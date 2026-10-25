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

    # The 2-1 leaf rule (fix 2-1 / fix 4-1) covers every value that is a leaf
    # at runtime, not only torch factories and move():
    #   * ``for p in model.parameters()/buffers()`` (and the ``named_*`` tuple
    #     forms) yields ``nn.Parameter``/registered buffer objects, which are
    #     leaves -- setting ``p.requires_grad = True`` on them is legal. Observed
    #     live on 2026-10-25 (agent_20261025_093522.log): a legal
    #     ``for param in ref_model.parameters(): param.requires_grad = True`` was
    #     rejected and cost a full regeneration.
    #   * ``t.detach()`` returns a new tensor with no graph, so the result is a
    #     leaf and also an independent differentiation root.
    # Both keep move()'s fail-closed guard: a locally rebound helper name (or a
    # monkeypatched Tensor method) voids recognition instead of trusting it.
    STATE_ITERATORS = {
        "parameters": 1, "buffers": 1, "named_parameters": 2, "named_buffers": 2,
    }
    state_iteration_shadowed = False
    detach_shadowed = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in STATE_ITERATORS:
                state_iteration_shadowed = True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute):
                    if target.attr in STATE_ITERATORS:
                        state_iteration_shadowed = True
                    if target.attr == "detach":
                        detach_shadowed = True

    def state_iteration_leaf(stmt):
        """Loop-variable names bound to model state by ``for p in m.parameters()``."""
        if state_iteration_shadowed:
            return set()
        value = stmt.iter
        if not (isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)):
            return set()
        arity = STATE_ITERATORS.get(value.func.attr)
        if arity is None:
            return set()
        names = [n.id for n in ast.walk(stmt.target) if isinstance(n, ast.Name)]
        # ``for p in m.parameters()`` binds the value directly; the named_*
        # forms bind ``(name, value)`` and the value is the leaf.
        if len(names) != arity:
            return set()
        return {names[-1]}

    def detach_leaf(value):
        """True when ``value`` is ``x.detach()``: a fresh leaf, no autograd graph."""
        if detach_shadowed:
            return False
        return (isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "detach"
                and not value.args and not value.keywords)

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
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                # The loop variable names leaves for the duration of the loop and
                # after it (it still holds the last yielded value), which is what
                # the 2-1 rule needs for ``for p in model.parameters()``.
                iterated = state_iteration_leaf(stmt)
                if iterated:
                    leaf_tensors.update(iterated)
                    move_tensors.difference_update(iterated)
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
                                "(a torch factory result, a value iterated from "
                                "Module.parameters()/buffers(), a .detach() result, "
                                "or a move(...) result from kernel_case_suite, which "
                                "always returns fresh leaf "
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
                    detached = detach_leaf(value)
                    factory_grad = factory and any(
                        k.arg == "requires_grad" and isinstance(k.value, ast.Constant)
                        and k.value.value is True for k in value.keywords)
                    # Detached targets in an otherwise differentiable loss are legal.
                    # Recognized factories also do not inherit their template's graph.
                    # A move() result is a fresh differentiation root: its copy_ runs
                    # under no_grad, so gradients never flow back to the source and it
                    # must never inherit "blocked" status (fix 4-1). A ``.detach()``
                    # result is the same kind of root: it carries no graph, so
                    # differentiating it is legal even when its source was computed
                    # under no_grad/inference_mode.
                    if not leaf and not factory and not moved and not detached and (
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
                    # torch-factory, .detach() or move()-derived value is a leaf;
                    # any other assignment revokes it. move()-derived names are
                    # tracked separately so their container elements count as
                    # leaves too.
                    for t in targets:
                        if isinstance(t, ast.Name):
                            if factory or detached:
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


# Backward-verification calls that a forward-only test must not contain.
_FORWARD_ONLY_BACKWARD_ATTRS = ("backward",)
_FORWARD_ONLY_GRAD_CALLS = (
    "torch.autograd.grad",
    "autograd.grad",
    "torch.autograd.backward",
)


def validate_forward_only_test(source: str) -> None:
    """Reject training-style grad checks in a forward-only correctness test.

    KernelBench scores the forward output only, so a test that also calls
    ``backward()`` reimposes a contract the benchmark never runs (observed as
    repeated "Weight gradient mismatch" after the forward case had passed).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Invalid correctness test: invalid Python: {exc}") from exc

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        if isinstance(node, ast.Attribute):
            inner = dotted(node.value)
            return f"{inner}.{node.attr}" if inner else node.attr
        if isinstance(node, ast.Name):
            return node.id
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if dotted(node.func) in _FORWARD_ONLY_GRAD_CALLS:
                raise ValueError(
                    "Invalid correctness test: forward-only verification was "
                    f"requested, but the test calls {dotted(node.func)} at line "
                    f"{node.lineno}. Remove every backward/autograd.grad step and "
                    "compare forward outputs only."
                )
            for keyword in node.keywords:
                if (
                    keyword.arg == "grad_outputs"
                    or (keyword.arg == "gradients")
                ):
                    raise ValueError(
                        "Invalid correctness test: forward-only verification was "
                        f"requested, but the test passes {keyword.arg}= at line "
                        f"{node.lineno}. Remove the gradient comparison."
                    )
        elif isinstance(node, ast.Attribute) and node.attr in _FORWARD_ONLY_BACKWARD_ATTRS:
            raise ValueError(
                "Invalid correctness test: forward-only verification was "
                f"requested, but the test calls .backward() at line {node.lineno}. "
                "Remove every backward step and compare forward outputs only."
            )
