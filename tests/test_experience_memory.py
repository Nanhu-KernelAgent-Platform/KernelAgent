from triton_kernel_agent.experience import (
    ExperienceRecord,
    SQLiteExperienceStore,
    build_operator_signature,
    format_experience_context,
    format_verified_seed_context,
)
def _problem(shape: tuple[int, int] = (1024, 4096)) -> str:
    return f"""
import torch
M, N = {shape}
class Model:
    def forward(self, x):
        return torch.rms_norm(x, (N,))
def get_inputs():
    return [torch.randn((M, N), dtype=torch.bfloat16)]
"""


def test_signature_extracts_operator_dtype_and_shape():
    signature = build_operator_signature(_problem())
    assert signature.semantic_type == "rmsnorm"
    assert "bfloat16" in signature.dtypes
    assert (1024, 4096) in signature.shapes
    assert signature.digest


def test_store_deduplicates_and_retrieves_similar_experience(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    record = ExperienceRecord(
        signature=signature,
        platform="musa",
        kernel_backend="musa",
        outcome="improved",
        kernel_code="FILE: kernel.mu\n```cpp\n// vectorized load\n```",
        verified=True,
        action="Vectorize contiguous loads",
        bottleneck="memory",
        improvement_pct=18.5,
    )
    assert store.add(record)
    assert not store.add(record)

    matches = store.search(
        build_operator_signature(_problem((2048, 4096))),
        platform="musa",
        kernel_backend="musa",
    )
    assert len(matches) == 1
    assert matches[0].action == "Vectorize contiguous loads"


def test_store_separates_platforms_and_formats_failures(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    store.add(
        ExperienceRecord(
            signature=signature,
            platform="cuda",
            kernel_backend="triton",
            outcome="success",
            kernel_code="def kernel_function(): pass",
            verified=True,
        )
    )
    failure = ExperienceRecord(
        signature=signature,
        platform="musa",
        kernel_backend="musa",
        outcome="failed",
        kernel_code="bad source",
        verified=False,
        action="Use unsupported intrinsic",
        error_message="compiler rejected intrinsic",
    )
    store.add(failure)

    matches = store.search(signature, "musa", "musa", include_failures=True)
    assert [item.id for item in matches] == [failure.id]
    context = format_experience_context(matches)
    assert "FAILED/REGRESSED" in context
    assert "compiler rejected intrinsic" in context
    assert "bad source" not in context


def test_verified_seed_library_requires_compatible_verified_source(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    success = ExperienceRecord(
        signature=signature,
        platform="musa",
        kernel_backend="musa",
        outcome="improved",
        kernel_code="FILE: kernel.mu\n```cpp\n// verified seed\n```",
        verified=True,
        kernel_time_ms=0.1,
    )
    store.add(success)
    store.add(
        ExperienceRecord(
            signature=signature,
            platform="musa",
            kernel_backend="musa",
            outcome="failed",
            kernel_code="bad source",
            verified=False,
        )
    )

    seeds = store.search_verified_seeds(signature, "musa", "musa")
    assert [seed.id for seed in seeds] == [success.id]
    context = format_verified_seed_context(seeds)
    assert "verified seed" in context
    assert "bad source" not in context


def test_signature_extracts_symbolic_shapes_from_prose_and_code():
    signature = build_operator_signature(
        """Generate a fused matmul kernel for this problem:

import torch
batch_size = 1024
in_features = 4096
out_features = 2058

def get_inputs():
    return [torch.randn(batch_size, in_features, dtype=torch.bfloat16)]

def get_init_inputs():
    return [in_features, out_features]
"""
    )
    assert signature.semantic_type == "matmul"
    assert signature.dtypes == ("bfloat16",)
    assert (1024, 4096) in signature.shapes
    assert (4096, 2058) in signature.shapes


def test_signature_classifies_sigmoid():
    signature = build_operator_signature(
        "def forward(x):\n    return torch.sigmoid(x)\n"
    )
    assert signature.semantic_type == "sigmoid"
