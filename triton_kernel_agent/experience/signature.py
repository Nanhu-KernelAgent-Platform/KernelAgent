"""Deterministic operator signature extraction without external services."""

from __future__ import annotations

import ast
import hashlib
import json
import re

from .models import OperatorSignature

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


def _semantic_type(text: str) -> str:
    lowered = text.lower().replace(" ", "_")
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


def build_operator_signature(problem_description: str) -> OperatorSignature:
    text = problem_description or ""
    tree = _parse_problem_ast(text)
    shapes = set(_extract_shapes(tree) if tree is not None else ())
    shapes.update(_extract_shapes_from_text(text))
    shapes = tuple(sorted(shapes, key=lambda value: (len(value), value))[:12])

    dtypes = tuple(sorted({match.lower() for match in _DTYPE_RE.findall(text)}))
    tokens = tuple(
        sorted(
            {
                token.lower()
                for token in _TOKEN_RE.findall(text)
                if token.lower() not in _IGNORED_TOKENS
            }
        )[:80]
    )
    semantic_type = _semantic_type(text)
    normalized = {
        "semantic_type": semantic_type,
        "dtypes": dtypes,
        "shapes": shapes,
        "tokens": tokens,
    }
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return OperatorSignature(semantic_type, dtypes, shapes, tokens, digest)
