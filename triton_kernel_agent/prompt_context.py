"""Static problem analysis for compact, reliable LLM context."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from triton_kernel_agent.experience.signature import build_operator_signature


@dataclass(frozen=True)
class TensorInputSpec:
    factory: str
    shape: str = "unknown"
    dtype: str = "unknown"
    device: str = "unspecified"


@dataclass(frozen=True)
class ProblemContext:
    operator_type: str
    forward_expression: str = ""
    operations: tuple[str, ...] = ()
    inputs: tuple[TensorInputSpec, ...] = ()
    init_inputs: str = ""
    constants: dict[str, str] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    shapes: tuple[tuple[int, ...], ...] = ()

    def format_for_prompt(self, max_chars: int = 6000) -> str:
        lines = [
            "## Structured problem contract",
            f"Operator family: {self.operator_type}",
        ]
        if self.forward_expression:
            lines.append(f"Forward semantics: {self.forward_expression}")
        if self.operations:
            lines.append(f"Operations: {', '.join(self.operations)}")
        if self.inputs:
            lines.append("Inputs:")
            for index, item in enumerate(self.inputs):
                lines.append(
                    f"- input {index}: {item.factory}, shape={item.shape}, "
                    f"dtype={item.dtype}, device={item.device}"
                )
        if self.init_inputs:
            lines.append(f"Initialization inputs: {self.init_inputs}")
        if self.constants:
            values = ", ".join(
                f"{name}={value}" for name, value in sorted(self.constants.items())
            )
            lines.append(f"Relevant constants: {values}")
        if self.dtypes:
            lines.append(f"Observed dtypes: {', '.join(self.dtypes)}")
        if self.shapes:
            lines.append(f"Observed literal shapes: {self.shapes}")
        if self.constraints:
            lines.append("Explicit constraints:")
            lines.extend(f"- {constraint}" for constraint in self.constraints)
        return "\n".join(lines)[:max_chars]


class ProblemContextBuilder:
    """Extract a stable problem contract from source or mixed prose/source."""

    def build(self, problem_description: str) -> ProblemContext:
        signature = build_operator_signature(problem_description)
        tree = self._parse_best_effort(problem_description)
        if tree is None:
            return ProblemContext(
                operator_type=signature.semantic_type,
                dtypes=signature.dtypes,
                shapes=signature.shapes,
            )

        forward = self._find_function(tree, "forward")
        get_inputs = self._find_function(tree, "get_inputs")
        get_init_inputs = self._find_function(tree, "get_init_inputs")
        return ProblemContext(
            operator_type=signature.semantic_type,
            forward_expression=self._return_expression(forward),
            operations=self._operations(forward),
            inputs=self._input_specs(get_inputs),
            init_inputs=self._return_expression(get_init_inputs),
            constants=self._constants(tree),
            constraints=self._constraints(tree),
            dtypes=signature.dtypes,
            shapes=signature.shapes,
        )

    @staticmethod
    def _parse_best_effort(text: str) -> ast.Module | None:
        candidates = [text]
        candidates.extend(
            match.group(1)
            for match in re.finditer(
                r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
            )
        )
        starts = [
            position
            for marker in ("import ", "from ", "class ", "def ")
            if (position := text.find(marker)) >= 0
        ]
        if starts:
            candidates.append(text[min(starts) :])
        for candidate in candidates:
            try:
                return ast.parse(candidate)
            except SyntaxError:
                continue
        return None

    @staticmethod
    def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    @staticmethod
    def _return_expression(node: ast.FunctionDef | None) -> str:
        if node is None:
            return ""
        returns = [item for item in ast.walk(node) if isinstance(item, ast.Return)]
        if not returns or returns[-1].value is None:
            return ""
        return ast.unparse(returns[-1].value)[:1200]

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        try:
            return ast.unparse(node.func)
        except Exception:
            return "call"

    def _operations(self, node: ast.FunctionDef | None) -> tuple[str, ...]:
        if node is None:
            return ()
        operations = {
            self._call_name(item)
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
        }
        return tuple(sorted(operations))[:30]

    def _input_specs(
        self, node: ast.FunctionDef | None
    ) -> tuple[TensorInputSpec, ...]:
        if node is None:
            return ()
        specs: list[TensorInputSpec] = []
        for item in ast.walk(node):
            if not isinstance(item, ast.Call):
                continue
            factory = self._call_name(item)
            if not any(
                name in factory
                for name in ("rand", "randn", "empty", "zeros", "ones", "tensor")
            ):
                continue
            shape = ast.unparse(item.args[0]) if item.args else "unknown"
            keywords = {keyword.arg: keyword.value for keyword in item.keywords}
            dtype = ast.unparse(keywords["dtype"]) if "dtype" in keywords else "unknown"
            device = (
                ast.unparse(keywords["device"])
                if "device" in keywords
                else "unspecified"
            )
            specs.append(TensorInputSpec(factory, shape, dtype, device))
        return tuple(specs[:20])

    @staticmethod
    def _constants(tree: ast.AST) -> dict[str, str]:
        constants: dict[str, str] = {}
        for node in getattr(tree, "body", []):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    rendered = ast.unparse(value)
                    if len(rendered) <= 120:
                        constants[target.id] = rendered
        return dict(list(constants.items())[:30])

    @staticmethod
    def _constraints(tree: ast.AST) -> tuple[str, ...]:
        constraints = [
            ast.unparse(node.test)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
        ]
        return tuple(constraints[:20])
