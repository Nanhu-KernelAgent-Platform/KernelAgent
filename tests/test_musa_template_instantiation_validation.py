"""Static check: explicit template instantiations must repeat their template.

Problem 62 (2026-10-25) generated ``template <typename scalar_t> void
launch_conv2d_forward(...)`` with 19 parameters but
``template void launch_conv2d_forward<float>(...)`` with 18 arguments. mcc could
only say "does not refer to a function template" plus a "could not match ...
against ..." note, so the refinement loop repeated the same off-by-one: two
29-minute refinements were lost.

``validate_template_instantiations`` reports the exact counts instead. It is
fail-open: any arity it cannot pin down is never reported.
"""

import pytest

from triton_kernel_agent.kernel_backend import (
    extract_kernel_bundle,
    validate_template_instantiations,
)
from triton_kernel_agent.worker import VerificationWorker


# --- problem 62 evidence --------------------------------------------------

# Real round_2 kernel.mu (declaration at line 48, instantiations at 216-221).
P62_ROUND_2 = '''
template <typename scalar_t>
void launch_conv2d_forward(
    const scalar_t* input, const scalar_t* weight, scalar_t* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int height_out, int width_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups) {
    (void)input;
}

template void launch_conv2d_forward<float>(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int);
template void launch_conv2d_forward<double>(const double*, const double*, double*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int);
'''

# The same declaration with a matching instantiation (19 arguments).
P62_FIXED = P62_ROUND_2.replace(
    "const float*, const float*, float*,"
    " int, int, int, int, int, int, int, int, int, int, int, int, int, int, int",
    "const float*, const float*, float*,"
    " int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int",
).replace(
    "const double*, const double*, double*,"
    " int, int, int, int, int, int, int, int, int, int, int, int, int, int, int",
    "const double*, const double*, double*,"
    " int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int",
)

NO_INSTANTIATIONS = '''
template <typename scalar_t>
void launch_add(const scalar_t* a, const scalar_t* b, scalar_t* out, int n) {
    (void)a; (void)b; (void)out; (void)n;
}
'''

MATCHING_INSTANTIATIONS = '''
template <typename scalar_t>
void launch_add(const scalar_t* a, const scalar_t* b, scalar_t* out, int n) {
    (void)a; (void)b; (void)out; (void)n;
}

template void launch_add<float>(const float*, const float*, float*, int);
template void launch_add<double>(const double*, const double*, double*, int);
'''

DEFAULTED_PARAMETER = '''
template <typename scalar_t>
void launch_pad(const scalar_t* in, scalar_t* out, int n, int pad = 0) {
    (void)in; (void)out; (void)n; (void)pad;
}

template void launch_pad<float>(const float*, float*, int);
'''

VARIADIC_TEMPLATE = '''
template <typename scalar_t, typename... Rest>
void launch_any(const scalar_t* in, Rest... rest) {
    (void)in;
}

template void launch_any<float>(const float*, int);
'''

VARIADIC_PARAMETERS = '''
template <typename scalar_t>
void launch_log(const scalar_t* in, int n, ...) {
    (void)in; (void)n;
}

template void launch_log<float>(const float*, int);
'''

CONFLICTING_ARITIES = '''
template <typename scalar_t>
void launch_scale(const scalar_t* in, scalar_t* out, int n) {
    (void)in; (void)out; (void)n;
}

template <typename scalar_t>
void launch_scale(const scalar_t* in, scalar_t* out, int n, int scale) {
    (void)in; (void)out; (void)n; (void)scale;
}

template void launch_scale<float>(const float*, float*, int);
'''

EXPLICIT_SPECIALIZATION = '''
template <typename scalar_t>
void launch_cast(const scalar_t* in, scalar_t* out) {
    (void)in; (void)out;
}

template <>
void launch_cast<int>(const int* in, int* out) {
    (void)in; (void)out;
}

template void launch_cast<float>(const float*, float*);
'''

CONDITIONAL_COMPILATION = '''
#if defined(__MUSA_ARCH__)
template <typename scalar_t>
void launch_cond(const scalar_t* in, int n) { (void)in; (void)n; }
template void launch_cond<float>(const float*, int);
#endif
'''

NESTED_TEMPLATE_CONSISTENT = '''
template <typename scalar_t>
void launch_pair(const scalar_t* in, std::pair<int, int> dims, int n) {
    (void)in; (void)dims; (void)n;
}

template void launch_pair<float>(const float*, std::pair<int, int>, int);
'''

NESTED_TEMPLATE_MISMATCH = '''
template <typename scalar_t>
void launch_pair(const scalar_t* in, std::pair<int, int> dims, int n) {
    (void)in; (void)dims; (void)n;
}

template void launch_pair<float>(const float*, std::pair<int, int>);
'''


def test_problem_62_round_2_is_rejected_with_exact_counts():
    reason = validate_template_instantiations({"kernel.mu": P62_ROUND_2})
    assert reason
    assert "kernel.mu:" in reason
    assert "launch_conv2d_forward" in reason
    assert "passes 18 arguments" in reason
    assert "declares 19 parameters" in reason
    # The repair payload must offer a concrete way out.
    assert "Repeat every template parameter" in reason
    assert "delete that line" in reason


def test_matching_instantiation_is_accepted():
    assert validate_template_instantiations({"kernel.mu": P62_FIXED}) is None


def test_kernel_without_explicit_instantiations_is_accepted():
    assert validate_template_instantiations({"kernel.mu": NO_INSTANTIATIONS}) is None


def test_matching_float_and_double_instantiations_are_accepted():
    assert validate_template_instantiations({"kernel.mu": MATCHING_INSTANTIATIONS}) is None


def test_defaulted_parameter_fails_open():
    # An instantiation may legally omit trailing default arguments.
    assert validate_template_instantiations({"kernel.mu": DEFAULTED_PARAMETER}) is None


def test_template_parameter_pack_fails_open():
    assert validate_template_instantiations({"kernel.mu": VARIADIC_TEMPLATE}) is None


def test_c_variadic_parameter_fails_open():
    assert validate_template_instantiations({"kernel.mu": VARIADIC_PARAMETERS}) is None


def test_conflicting_declarations_fail_open():
    # Two declarations with different arities: never guess which one applies.
    assert validate_template_instantiations({"kernel.mu": CONFLICTING_ARITIES}) is None


def test_explicit_specialization_fails_open():
    assert validate_template_instantiations({"kernel.mu": EXPLICIT_SPECIALIZATION}) is None


def test_conditional_compilation_fails_open():
    assert validate_template_instantiations({"kernel.mu": CONDITIONAL_COMPILATION}) is None


def test_nested_template_arguments_are_counted_not_split():
    # ``std::pair<int, int>`` is ONE parameter; a naive comma split says two.
    assert validate_template_instantiations({"kernel.mu": NESTED_TEMPLATE_CONSISTENT}) is None
    reason = validate_template_instantiations({"kernel.mu": NESTED_TEMPLATE_MISMATCH})
    assert reason and "passes 2 arguments" in reason and "declares 3 parameters" in reason


def test_unparseable_inputs_are_accepted():
    assert validate_template_instantiations("") is None
    assert validate_template_instantiations("def kernel_function(x): return x") is None
    assert validate_template_instantiations({"kernel.py": "def kernel_function(x): pass"}) is None


def _bundle_text(kernel_mu: str) -> str:
    return (
        "FILE: kernel.py\n```python\ndef kernel_function(x): return ext.forward(x)\n```\n\n"
        "FILE: binding.cpp\n```cpp\n#include <torch/extension.h>\n```\n\n"
        "FILE: kernel.mu\n```cpp\n" + kernel_mu + "\n```\n\n"
        "FILE: setup.py\n```python\n"
        "from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
        "from setuptools import setup\n"
        "setup(name='extension', ext_modules=[MUSAExtension(name='extension', "
        "sources=['binding.cpp', 'kernel.mu'])], "
        "cmdclass={'build_ext': BuildExtension})\n```\n"
    )


def test_extracted_bundle_is_checked():
    bundle = extract_kernel_bundle(_bundle_text(P62_ROUND_2))
    reason = validate_template_instantiations(bundle)
    assert reason and "launch_conv2d_forward" in reason
    good = extract_kernel_bundle(_bundle_text(P62_FIXED))
    assert validate_template_instantiations(good) is None


def _worker():
    worker = VerificationWorker.__new__(VerificationWorker)
    worker.logger = type("L", (), {"error": lambda *a, **k: None,
                                   "warning": lambda *a, **k: None,
                                   "info": lambda *a, **k: None})()
    from triton_kernel_agent.kernel_backend import get_kernel_backend

    worker._kernel_backend = get_kernel_backend("musa")
    worker.last_case_results = []
    worker.test_files = []
    worker._detect_pytorch_compute = lambda code: None
    return worker


def test_verification_pass_reports_the_arity_mismatch_before_running_tests():
    worker = _worker()
    worker._run_test = lambda: pytest.fail("the build must not run")
    bundle = extract_kernel_bundle(_bundle_text(P62_ROUND_2))

    success, stdout, stderr, violation = worker._single_verification_pass(bundle)

    assert success is False
    assert stdout == ""
    assert "launch_conv2d_forward" in stderr
    assert violation is None


def test_verification_pass_defers_to_the_tests_when_arity_matches():
    worker = _worker()
    worker._run_test = lambda: (True, "ok", "")
    bundle = extract_kernel_bundle(_bundle_text(P62_FIXED))

    assert worker._single_verification_pass(bundle) == (True, "ok", "", None)
