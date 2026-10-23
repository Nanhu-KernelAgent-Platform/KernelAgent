"""Bounded static checks for a correctness test, never a correctness proof.

Reject a locally visible disabled-grad path later used for differentiation.
Do not silently edit tests, discard failures, or infer that a kernel is correct.
Dynamic/helper-mediated autograd errors still require runtime investigation.
"""

import ast


def validate_kernel_globals(source: str) -> None:
    """Reject visible framework monkeypatches; not a sandbox/security proof."""
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    aliases = {}
    def dotted(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return dotted(node.value) + '.' + node.attr
        if isinstance(node, ast.Subscript):
            return dotted(node.value)
        if isinstance(node, ast.Call) and dotted(node.func) in {'getattr', 'vars'} and node.args:
            return dotted(node.args[0])
        return ''
    def protected(path):
        return path.split('.')[0] in {'torch', 'torch_musa', 'builtins', '__builtins__', 'sys'}
    for node in nodes:
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or '') + '.' + item.name
    # Resolve direct local aliases without depending on AST traversal order.
    for _ in range(4):
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value:
                path = dotted(node.value)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and protected(path):
                        aliases[target.id] = path
    for node in nodes:
        targets = (node.targets if isinstance(node, (ast.Assign, ast.Delete)) else
                   [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
        forbidden = any(not isinstance(target, ast.Name) and protected(dotted(target)) for target in targets)
        if isinstance(node, ast.Call):
            name = dotted(node.func)
            forbidden |= (name in {'setattr', 'delattr', 'builtins.setattr', 'builtins.delattr'}
                          and bool(node.args) and protected(dotted(node.args[0])))
            forbidden |= protected(name) and ('.__dict__.' in name or name.endswith('.__setattr__'))
            if name.endswith('.patch') or name.endswith('.patch.object'):
                forbidden |= bool(node.args) and (protected(dotted(node.args[0])) or
                    isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                    and protected(node.args[0].value))
        if forbidden:
            raise ValueError(f'Kernel must not modify framework/global verification behavior at line {node.lineno}')


def validate_test_layout(source: str) -> None:
    """Factory input device copies must use the shared stride-preserving mover."""
    tree = ast.parse(source)
    for scope in [tree, *[node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]]:
        nodes = [node for stmt in scope.body if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                 for node in ast.walk(stmt)]
        inputs = set()
        def derived(value):
            if isinstance(value, ast.Name): return value.id in inputs
            if isinstance(value, ast.Subscript): return derived(value.value)
            if isinstance(value, ast.Call):
                if isinstance(value.func, ast.Name) and value.func.id == 'get_inputs': return True
                if isinstance(value.func, ast.Attribute) and value.func.attr in {'to', 'cuda', 'cpu', 'clone', 'detach', 'requires_grad_'}:
                    return derived(value.func.value)
            return False
        for _ in range(4):
            for node in nodes:
                if isinstance(node, ast.comprehension) and derived(node.iter):
                    inputs.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and derived(node.value):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    inputs.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
        for node in nodes:
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {'to', 'cuda', 'cpu'} and derived(node.func.value)):
                raise ValueError('Invalid correctness test: factory tensor .to()/.cuda() can change strides. '
                                 'Use from kernel_case_suite import move; move(inputs, "musa"). '
                                 'Repair the test, not the kernel.')


def validate_test_autograd(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Invalid correctness test: invalid Python: {exc}") from exc

    # Fix 4-1: import alias resolution so the grad-mode checks and the 2-1
    # leaf rule see through ``import torch as t`` and
    # ``from kernel_case_suite import move as mv`` (mirrors
    # validate_kernel_globals' dotted()). Later imports win, matching the
    # runtime rebinding semantics of Python.
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = (
                    item.name if item.asname else item.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or "") + "." + item.name

    def dotted(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return dotted(node.value) + "." + node.attr
        return ""

    def dependencies(node):
        # Shape/dtype/device metadata and scalar shape queries do not carry gradients.
        if isinstance(node, ast.Attribute) and node.attr in ("shape", "dtype", "device", "ndim"):
            return set()
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("numel", "size", "stride", "dim", "nelement")):
            return set()
        if isinstance(node, ast.Name):
            return {node.id}
        return set().union(*(dependencies(child) for child in ast.iter_child_nodes(node)))

    # Names whose most recent assignment is a direct torch factory call.
    # A tensor created by a torch factory is a fresh leaf, so setting
    # ``X.requires_grad = True`` on it is valid. Any derived tensor (e.g. a
    # ``.clone()`` result) is not a leaf, and setting the attribute would fail
    # at runtime -- flag it statically instead of failing 8 refinement rounds.
    FACTORY_CALLS = {
        "torch.randn", "torch.rand", "torch.randn_like", "torch.rand_like",
        "torch.zeros", "torch.ones", "torch.empty", "torch.full",
        "torch.zeros_like", "torch.ones_like", "torch.empty_like",
        "torch.arange", "torch.tensor",
    }

    # Fix 4-1: the suite's move() always returns fresh leaf tensors
    # (case_suite.py: empty_strided allocation + no_grad copy_ +
    # requires_grad_()), and maps list/tuple/dict recursively, so every
    # element is also a fresh leaf. A locally rebound name (def move /
    # move = ...) or a module attribute rebind (kernel_case_suite.move =
    # ...) voids that guarantee: the name may run arbitrary user code, so
    # recognition must fail closed. Tests are not covered by
    # validate_kernel_globals' framework-patch protection (that runs on
    # kernel.py and protects only torch/torch_musa), hence the guard here.
    MOVE = "kernel_case_suite.move"
    rebound = set()
    move_shadowed = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rebound.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and dotted(target) == MOVE:
                    move_shadowed = True
                rebound.update(
                    n.id for n in ast.walk(target) if isinstance(n, ast.Name))

    def move_leaf(value, move_tensors):
        """True when ``value`` is a fresh-leaf-producing suite move() result.

        Covers the direct call, subscript chains over the call (move maps
        containers recursively), and subscripting a name previously bound to
        a move() result (``inputs = move(...); x = inputs[0]``) -- the
        P45 attempt-2 form. The chain stops at anything that is not a
        subscript or the move call itself: ``move(...)[0].clone()`` is a
        derived tensor and stays rejected.
        """
        if move_shadowed:
            return False
        while isinstance(value, ast.Subscript):
            value = value.value
            if isinstance(value, ast.Name) and value.id in move_tensors:
                return True
        if not (isinstance(value, ast.Call) and dotted(value.func) == MOVE):
            return False
        base = value.func
        while isinstance(base, ast.Attribute):
            base = base.value
        return not (isinstance(base, ast.Name) and base.id in rebound)

    def check_block(statements, blocked, live, leaf_tensors, move_tensors, disabled=False):
        for stmt in statements:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                check_block(stmt.body, set(), set(), set(), set())
                continue
            if isinstance(stmt, ast.ClassDef):
                check_block(stmt.body, set(), set(), set(), set())
                continue
            if isinstance(stmt, (ast.With, ast.AsyncWith)):
                mode = disabled
                for item in stmt.items:
                    expr = item.context_expr
                    if isinstance(expr, ast.Call):
                        name = dotted(expr.func)
                        if name in ("torch.no_grad", "torch.inference_mode"):
                            mode = True
                        elif name == "torch.enable_grad":
                            mode = False
                check_block(stmt.body, blocked, live, leaf_tensors, move_tensors, mode)
                continue
            # Scan this statement's expressions, not nested function/control bodies.
            expressions = [child for child in ast.iter_child_nodes(stmt)
                           if isinstance(child, ast.expr)]
            for expr in expressions:
                for call in ast.walk(expr):
                    if not isinstance(call, ast.Call):
                        continue
                    target = None
                    if isinstance(call.func, ast.Attribute) and call.func.attr == "backward":
                        target = call.func.value
                    elif dotted(call.func) in ("torch.autograd.grad", "torch.autograd.backward"):
                        target = call.args[0] if call.args else next(
                            (k.value for k in call.keywords if k.arg in ("outputs", "tensors")), None)
                    names = dependencies(target) if target is not None else set()
                    if names & blocked and not names & live:
                        raise ValueError(
                            "Invalid correctness test: backward/autograd uses a value "
                            f"derived from no_grad/inference_mode at line {call.lineno}. "
                            "Recompute the differentiable reference with torch.enable_grad(), "
                            "using independent leaf inputs; repair the test, not the kernel."
                        )
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                value = stmt.value
                if value is not None:
                    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                    # 2-1 rule: reject ``X.requires_grad = True`` when ``X`` has not
                    # been created as a fresh leaf tensor (torch factory or a
                    # kernel_case_suite.move() result, which always returns
                    # fresh leaves -- fix 4-1).
                    for t in targets:
                        if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                and t.attr == "requires_grad"
                                and isinstance(value, ast.Constant) and value.value is True
                                and t.value.id not in leaf_tensors):
                            raise ValueError(
                                "Invalid correctness test: only a fresh leaf tensor "
                                "(a torch factory result, or a move(...) result from "
                                "kernel_case_suite, which always returns fresh leaf "
                                "tensors) may have .requires_grad set to True, not a "
                                f"derived tensor at line {t.lineno}. A value such as "
                                ".clone() is not a leaf and fails at runtime; to "
                                "enable grad on an existing fresh leaf, call "
                                ".requires_grad_() instead of assigning the attribute. "
                                "Repair the test, not the kernel."
                            )
                    names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
                    value_dependencies = dependencies(value)
                    # Restoring a leaf is not restoring the detached reference's derivatives.
                    # Let runtime semantic checks judge explicit leaf construction instead.
                    leaf = (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                            and value.func.attr == "requires_grad_"
                            and (not value.args or not isinstance(value.args[0], ast.Constant)
                                 or value.args[0].value is True))
                    factory = (isinstance(value, ast.Call) and dotted(value.func) in FACTORY_CALLS)
                    moved = move_leaf(value, move_tensors)
                    factory_grad = factory and any(
                        k.arg == "requires_grad" and isinstance(k.value, ast.Constant)
                        and k.value.value is True for k in value.keywords)
                    # Detached targets in an otherwise differentiable loss are legal.
                    # Recognized factories also do not inherit their template's graph.
                    # A move() result is a fresh differentiation root: its copy_ runs
                    # under no_grad, so gradients never flow back to the source and it
                    # must never inherit "blocked" status (fix 4-1).
                    if not leaf and not factory and not moved and (
                        disabled or (value_dependencies & blocked and not value_dependencies & live)
                    ):
                        blocked.update(names)
                        live.difference_update(names)
                    else:
                        blocked.difference_update(names)
                        if leaf or factory_grad or (not disabled and (
                            value_dependencies & live or (isinstance(value, ast.Call) and not factory)
                        )):
                            live.update(names)
                        else:
                            live.difference_update(names)
                    # Track leaf status for the 2-1 rule: a plain name assigned a
                    # torch-factory or move()-derived value is a leaf; any other
                    # assignment revokes it. move()-derived names are tracked
                    # separately so their container elements count as leaves too.
                    for t in targets:
                        if isinstance(t, ast.Name):
                            if factory:
                                leaf_tensors.add(t.id)
                                move_tensors.discard(t.id)
                            elif moved:
                                leaf_tensors.add(t.id)
                                move_tensors.add(t.id)
                            else:
                                leaf_tensors.discard(t.id)
                                move_tensors.discard(t.id)
            # Branch-local assignments must not erase another possible blocked path.
            for field in ("body", "orelse", "finalbody"):
                body = getattr(stmt, field, None)
                if isinstance(body, list):
                    branch = set(blocked)
                    branch_live = set(live)
                    branch_leaves = set(leaf_tensors)
                    branch_moved = set(move_tensors)
                    check_block(body, branch, branch_live, branch_leaves,
                                branch_moved, disabled)
                    blocked.update(branch)
                    live.intersection_update(branch_live)
                    leaf_tensors.update(branch_leaves)
                    move_tensors.update(branch_moved)
            for handler in getattr(stmt, "handlers", []):
                branch = set(blocked)
                branch_live = set(live)
                branch_leaves = set(leaf_tensors)
                branch_moved = set(move_tensors)
                check_block(handler.body, branch, branch_live, branch_leaves,
                            branch_moved, disabled)
                blocked.update(branch)
                live.intersection_update(branch_live)
                leaf_tensors.update(branch_leaves)
                move_tensors.update(branch_moved)

    check_block(tree.body, set(), set(), set(), set())


def validate_test_kernel_call_binding(source: str) -> None:
    """The main test is generated before the kernel, so it must not guess a signature.

    Fix 3-1 (docs/design_kernelagent_fix_3-1_signature_contract_binding.md):
    every consumer of ``kernel_function`` — the deterministic case suite, the
    KernelBench evaluator, and the generated main test — resolves parameters
    through ``bind_kernel_call`` (by name against the reference model's state,
    then positionally from ``get_inputs()``). A hardcoded ``kernel_function(...)``
    call in the main test is a guess about a signature that has not been
    generated yet; P50/P56 burned full verification runs on exactly that
    mismatch. Reject the direct call statically and require at least one
    ``bind_kernel_call`` call site instead.
    """
    tree = ast.parse(source)

    # Import alias resolution mirrors validate_kernel_globals' dotted() pattern:
    # ``from kernel import kernel_function as kf`` and ``import kernel as k``
    # both must resolve to "kernel.kernel_function".
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = (
                    item.name if item.asname else item.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or "") + "." + item.name

    def dotted(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return dotted(node.value) + "." + node.attr
        return ""

    KERNEL = "kernel.kernel_function"
    BIND = "kernel_case_suite.bind_kernel_call"

    def is_kernel_func(func):
        if isinstance(func, ast.Name):
            return func.id == "kernel_function" or aliases.get(func.id) == KERNEL
        return dotted(func) == KERNEL

    # An import of kernel_function already proves the test intends to exercise
    # the kernel, so the bind-call requirement applies from that point on.
    referenced_kernel = any(
        isinstance(node, ast.ImportFrom) and node.module == "kernel"
        and any((item.asname or item.name) == "kernel_function" for item in node.names)
        for node in ast.walk(tree)
    )
    bind_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if is_kernel_func(node.func):
                raise ValueError(
                    "Invalid correctness test: kernel_function must be invoked "
                    "through bind_kernel_call(kernel_function, model_copy, inputs) "
                    f"from kernel_case_suite, not called directly at line {node.lineno}; "
                    "the test is generated before the kernel, so a hardcoded call "
                    "signature is a guess that cannot be checked. "
                    "Repair the test, not the kernel."
                )
            if dotted(node.func) == BIND:
                # Only a properly imported bind_kernel_call counts as an
                # invocation site; a bare unresolved call is a missing import
                # and must fall through to the instructive error below.
                bind_calls += 1
        elif not referenced_kernel and (
            is_kernel_func(node) if isinstance(node, (ast.Name, ast.Attribute)) else False
        ):
            referenced_kernel = True
    if referenced_kernel and bind_calls == 0:
        raise ValueError(
            "Invalid correctness test: kernel_function is referenced but never "
            "exercised; import bind_kernel_call from kernel_case_suite and route "
            "the call through bind_kernel_call(kernel_function, model_copy, inputs)."
        )
