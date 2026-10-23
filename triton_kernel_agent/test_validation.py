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

    def dotted(node):
        if isinstance(node, ast.Name):
            return node.id
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

    def check_block(statements, blocked, live, leaf_tensors, disabled=False):
        for stmt in statements:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                check_block(stmt.body, set(), set(), set())
                continue
            if isinstance(stmt, ast.ClassDef):
                check_block(stmt.body, set(), set(), set())
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
                check_block(stmt.body, blocked, live, leaf_tensors, mode)
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
                    # been created as a fresh torch-factory leaf tensor.
                    for t in targets:
                        if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                and t.attr == "requires_grad"
                                and isinstance(value, ast.Constant) and value.value is True
                                and t.value.id not in leaf_tensors):
                            raise ValueError(
                                "Invalid correctness test: only a fresh leaf tensor "
                                "(e.g. torch.randn(...), or a factory with "
                                "requires_grad=True) may have .requires_grad set to True, "
                                f"not a derived tensor at line {t.lineno}. A value such as "
                                ".clone() is not a leaf and fails at runtime; "
                                "repair the test, not the kernel."
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
                    factory_grad = factory and any(
                        k.arg == "requires_grad" and isinstance(k.value, ast.Constant)
                        and k.value.value is True for k in value.keywords)
                    # Detached targets in an otherwise differentiable loss are legal.
                    # Recognized factories also do not inherit their template's graph.
                    if not leaf and not factory and (
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
                    # torch-factory value is a leaf; any other assignment revokes it.
                    for t in targets:
                        if isinstance(t, ast.Name):
                            if factory:
                                leaf_tensors.add(t.id)
                            else:
                                leaf_tensors.discard(t.id)
            # Branch-local assignments must not erase another possible blocked path.
            for field in ("body", "orelse", "finalbody"):
                body = getattr(stmt, field, None)
                if isinstance(body, list):
                    branch = set(blocked)
                    branch_live = set(live)
                    branch_leaves = set(leaf_tensors)
                    check_block(body, branch, branch_live, branch_leaves, disabled)
                    blocked.update(branch)
                    live.intersection_update(branch_live)
                    leaf_tensors.update(branch_leaves)
            for handler in getattr(stmt, "handlers", []):
                branch = set(blocked)
                branch_live = set(live)
                branch_leaves = set(leaf_tensors)
                check_block(handler.body, branch, branch_live, branch_leaves, disabled)
                blocked.update(branch)
                live.intersection_update(branch_live)
                leaf_tensors.update(branch_leaves)

    check_block(tree.body, set(), set(), set())
