"""Deterministic operator signature extraction without external services."""

from __future__ import annotations

import ast
import hashlib
import json
import re

from .models import OperatorSignature
from .graph_identity import trace_reference_graph


def declared_execution_direction(source: str) -> str | None:
    """Read an explicit target; never infer direction from operator text."""
    match = re.search(
        r'(?m)^KERNELAGENT_OPTIMIZATION_TARGET\s*=\s*["\'](forward|backward)["\']',
        source,
    )
    return match.group(1) if match else None

_DTYPE_RE = re.compile(
    r"(?:torch\.)?(bfloat16|float16|float32|float64|int8|int16|int32|"
    r"int64|bool|fp8|bf16|fp16|fp32)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_OPERATOR_KEYWORDS = (
    "group_gemm",
    "grouped_gemm",
    "rmsnorm",
    "layernorm",
    "softmax",
    "sigmoid",
    "transpose",
    "matmul",
    "gemm",
    "reduction",
    "reduce",
    "moe",
    "attention",
    "convolution",
    "conv",
    "pooling",
    "pool",
    "elementwise",
)
_IGNORED_TOKENS = {
    "torch",
    "return",
    "class",
    "model",
    "self",
    "input",
    "inputs",
    "tensor",
    "device",
    "dtype",
}


def _focused_text(text: str) -> tuple[str, bool]:
    # Descriptions may include the surrounding model. Prefer the explicit target.
    target = re.search(
        r"(?im)^[ \t]*(?:only this (?:block should be generated and optimized|part)"
        r"|only generate and optimize this block|target operator only):"
        r"[ \t]*(?:\n[ \t]*)?([^\n]+)$",
        text,
    )
    return (target.group(1), True) if target else (text, False)


def _reference_source(text: str) -> str:
    """Prefer the declared reference Model over surrounding application prose."""
    match = re.search(r"(?im)^##\s*Describe2\b[^\n]*\n", text)
    if not match:
        return text
    source = re.split(r"(?m)^##\s+", text[match.end():], maxsplit=1)[0].strip()
    fenced = re.fullmatch(r"```(?:python)?\s*\n([\s\S]*?)\n```", source)
    return fenced.group(1) if fenced else source


def reference_source_from_test(test_code: str) -> str | None:
    """Extract the reference executed by a verified test, excluding test harness.

    Only top-level definitions needed by Model/input factories are retained.
    In particular, importing the generated ``kernel`` is neither necessary nor
    valid when tracing the reference in a separate process.
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return None
    definitions: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Import):
            for alias in node.names:
                definitions[alias.asname or alias.name.split(".")[0]] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                definitions[alias.asname or alias.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        definitions[name.id] = node
    if "Model" not in definitions or "get_inputs" not in definitions:
        return None
    selected = {"Model", "get_inputs", "get_init_inputs", "get_test_cases",
                "KERNELAGENT_OPTIMIZATION_TARGET"} & definitions.keys()
    pending = list(selected)
    while pending:
        node = definitions[pending.pop()]
        for child in ast.walk(node):
            if (isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                    and child.id in definitions and child.id not in selected):
                selected.add(child.id)
                pending.append(child.id)
    nodes = {id(definitions[name]) for name in selected}
    retained = [node for node in tree.body if id(node) in nodes]
    for node in retained:
        if isinstance(node, ast.ImportFrom) and (
            node.module == "kernel" or (node.module or "").startswith("kernel.")
        ):
            return None
        if isinstance(node, ast.Import) and any(
            alias.name == "kernel" or alias.name.startswith("kernel.")
            for alias in node.names
        ):
            return None
    return ast.unparse(ast.Module(body=retained, type_ignores=[])) + "\n"


def _semantic_type(text: str) -> str:
    focused, _ = _focused_text(text)
    lowered = focused.lower().replace(" ", "_")
    compact = lowered.replace("_", "")
    if "rsqrt" in compact and ("variance" in compact or "mean(" in compact):
        return "rmsnorm"
    for keyword in _OPERATOR_KEYWORDS:
        if keyword in lowered or keyword.replace("_", "") in compact:
            aliases = {
                "grouped_gemm": "group_gemm",
                "gemm": "matmul",
                "reduce": "reduction",
                "conv": "convolution",
                "pool": "pooling",
            }
            return aliases.get(keyword, keyword)
    return "unknown"


def _extract_shapes(tree: ast.AST) -> tuple[tuple[int, ...], ...]:
    shapes: set[tuple[int, ...]] = set()
    constants: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, int):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and value.value > 0:
                constants[target.id] = value.value

    def resolve(element: ast.AST) -> int | None:
        if isinstance(element, ast.Constant) and isinstance(element.value, int):
            return element.value if element.value > 0 else None
        if isinstance(element, ast.Name):
            return constants.get(element.id)
        return None

    for node in ast.walk(tree):
        elements: list[ast.AST] | None = None
        if isinstance(node, (ast.Tuple, ast.List)):
            elements = list(node.elts)
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            elements = list(node.args)
        if not elements or len(elements) < 2:
            continue
        values = [resolve(element) for element in elements]
        if all(value is not None for value in values):
            shapes.add(tuple(values))
    return tuple(sorted(shapes, key=lambda value: (len(value), value))[:12])


def _parse_problem_ast(text: str) -> ast.AST | None:
    """Parse either plain Python or prose followed by a Python problem body."""

    try:
        return ast.parse(text)
    except SyntaxError:
        pass
    lines = text.splitlines()
    python_start = re.compile(r"^\s*(?:import\s|from\s|class\s|def\s|@)")
    for index, line in enumerate(lines):
        if not python_start.match(line):
            continue
        try:
            return ast.parse("\n".join(lines[index:]))
        except SyntaxError:
            continue
    return None


def _call_operator(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    name = node.func.attr.lower()
    if name.startswith("conv"):
        return "convolution"
    if name in {"matmul", "mm", "bmm"}:
        return "matmul"
    aliases = {
        "relu": "relu", "softmax": "softmax", "sigmoid": "sigmoid",
        "gelu": "gelu", "silu": "silu", "layer_norm": "layernorm",
        "layernorm": "layernorm", "rms_norm": "rmsnorm",
        "rmsnorm": "rmsnorm", "batch_norm": "batchnorm",
        "batchnorm": "batchnorm", "max_pool2d": "pooling",
        "avg_pool2d": "pooling",
    }
    return aliases.get(name)


def _extract_components(text: str) -> tuple[str, ...]:
    """Conservative directed two-node dataflow motifs from Python forward code."""
    tree = _parse_problem_ast(text)
    if tree is None:
        return ()
    forwards = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "forward"
    ]
    bodies = [node.body for node in forwards] or [getattr(tree, "body", [])]
    edges: set[str] = set()

    for body in bodies:
        producers: dict[str, str] = {}

        def producer(node: ast.AST) -> str | None:
            if isinstance(node, ast.Name):
                return producers.get(node.id)
            operator = _call_operator(node)
            if operator:
                return operator
            if isinstance(node, ast.BinOp):
                sources = {source for child in (node.left, node.right)
                           if (source := producer(child))}
                return next(iter(sources)) if len(sources) == 1 else None
            if isinstance(node, ast.UnaryOp):
                return producer(node.operand)
            return None

        for statement in body:
            value = getattr(statement, "value", None)
            if value is None:
                continue
            for call in ast.walk(value):
                consumer = _call_operator(call)
                if consumer is None:
                    continue
                for argument in (*call.args, *(kw.value for kw in call.keywords)):
                    source = producer(argument)
                    if source and source != consumer:
                        edges.add(f"{source}_{consumer}")
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        source = producer(value)
                        producers.pop(target.id, None)
                        if source:
                            producers[target.id] = source
    return tuple(sorted(edges))


def _extract_shapes_from_text(text: str) -> tuple[tuple[int, ...], ...]:
    """Best-effort shape extraction for prose or structurally invalid Python."""

    constants = {
        name: int(value)
        for name, value in re.findall(
            r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*([1-9]\d*)\s*(?:#.*)?$", text
        )
    }
    shapes: set[tuple[int, ...]] = set()

    def add_fields(raw: str) -> None:
        values: list[int] = []
        for field in raw.split(","):
            token = field.strip()
            if not token or "=" in token:
                break
            if token.isdigit() and int(token) > 0:
                values.append(int(token))
            elif token in constants:
                values.append(constants[token])
            else:
                break
        if len(values) >= 2:
            shapes.add(tuple(values))

    for raw in re.findall(
        r"(?:torch\.)?(?:randn|rand|empty|zeros|ones|full)\s*\(([^()\n]*)\)",
        text,
    ):
        add_fields(raw)
    for raw in re.findall(r"\[([^\[\]\n]+)\]", text):
        add_fields(raw)
    return tuple(sorted(shapes, key=lambda value: (len(value), value))[:12])


def build_operator_signature(
    problem_description: str, *, trace_graph: bool = False,
    execution_direction: str | None = None,
) -> OperatorSignature:
    text = problem_description or ""
    reference = _reference_source(text)
    tree = _parse_problem_ast(reference)
    shapes = set(_extract_shapes(tree) if tree is not None else ())
    shapes.update(_extract_shapes_from_text(reference))
    shapes = tuple(sorted(shapes, key=lambda value: (len(value), value))[:12])

    dtypes = tuple(sorted({match.lower() for match in _DTYPE_RE.findall(reference)}))
    tokens = tuple(
        sorted(
            {
                token.lower()
                for token in _TOKEN_RE.findall(reference)
                if token.lower() not in _IGNORED_TOKENS
            }
        )[:80]
    )
    semantic_type = _semantic_type(reference)
    focused, _ = _focused_text(reference)
    components = set(_extract_components(focused))
    components.discard(semantic_type)
    components = tuple(sorted(components))
    normalized = {
        "semantic_type": semantic_type,
        "components": components,
        "dtypes": dtypes,
        "shapes": shapes,
        "tokens": tokens,
    }
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    contract_digest = hashlib.sha256(
        re.sub(r"\s+", " ", text.strip()).encode("utf-8")
    ).hexdigest()[:24]
    if execution_direction not in {None, "forward", "backward", "joint", "unknown"}:
        raise ValueError("invalid graph execution direction")
    # Override only the probe's execution target, after extracting Describe2.
    # Prose/request fingerprints and code-seed contract gates stay unchanged.
    graph_source = reference
    if execution_direction in {"forward", "backward", "joint"}:
        graph_source += f"\nKERNELAGENT_OPTIMIZATION_TARGET = {execution_direction!r}\n"
    graph = trace_reference_graph(graph_source) if trace_graph and tree is not None else None
    return OperatorSignature(
        semantic_type, dtypes, shapes, tokens, digest, components,
        contract_digest,
        graph_version=graph["version"] if graph else "",
        graph_digest=graph["digest"] if graph else "",
        graph_nodes=tuple(graph["nodes"]) if graph else (),
        interface=graph["interface"] if graph else {},
        graph_tensor_specs=graph.get("tensor_specs", {}) if graph else {},
    )
