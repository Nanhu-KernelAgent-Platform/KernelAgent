"""Retrieve bounded MUSA guidance for the current operator family."""

from __future__ import annotations

from pathlib import Path


class MUSAKnowledgePack:
    _ALIASES = {
        "layernorm": "norm",
        "rmsnorm": "norm",
        "group_gemm": "moe",
        "grouped_gemm": "moe",
        "gemm": "matmul",
        "softmax": "reduction",
        "reduce": "reduction",
        "pooling": "reduction",
        "unknown": "common",
    }

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = (
            Path(root)
            if root is not None
            else Path(__file__).parent / "knowledge" / "musa"
        )

    def retrieve(self, operator_type: str, max_chars: int = 8000) -> str:
        family = self._ALIASES.get(operator_type, operator_type)
        names = ["common"]
        if family != "common" and (self.root / f"{family}.md").exists():
            names.append(family)
        sections: list[str] = ["## MUSA knowledge relevant to this operator"]
        for name in names:
            path = self.root / f"{name}.md"
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8").strip()
            if len("\n\n".join(sections + [content])) > max_chars:
                break
            sections.append(content)
        return "\n\n".join(sections) if len(sections) > 1 else ""
