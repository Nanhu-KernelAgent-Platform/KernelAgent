#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end BF16 matmul+sigmoid test harness."""

import argparse
import os
import sys
import time
from dotenv import load_dotenv
from triton_kernel_agent import TritonKernelAgent
from triton_kernel_agent.platform_config import get_platform
from utils.providers.openai_base import OPENAI_REASONING_EFFORTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-backend", choices=("triton", "musa"), default="triton")
    parser.add_argument("--target-platform", choices=("cuda", "musa", "xpu"), default="cuda")
    parser.add_argument(
        "--reasoning-effort",
        choices=OPENAI_REASONING_EFFORTS,
        default=None,
        help="GPT-5.6 reasoning effort; overrides OPENAI_REASONING_EFFORT",
    )
    return parser.parse_args()


def main():
    """Generate and test a BF16 matmul kernel with fused sigmoid activation."""
    # Load environment
    load_dotenv()

    # Create agent
    args = parse_args()
    if args.reasoning_effort:
        os.environ["OPENAI_REASONING_EFFORT"] = args.reasoning_effort
    print(
        "Reasoning effort: "
        f"{os.getenv('OPENAI_REASONING_EFFORT', 'provider default')}"
    )
    agent = TritonKernelAgent(
        kernel_backend=args.kernel_backend,
        target_platform=get_platform(args.target_platform),
        test_timeout_s=180 if args.kernel_backend == "musa" else 30,
        max_rounds=2,
    )

    print("=" * 80)
    print("BF16 Matmul with Fused Sigmoid Activation")
    print("Matrix dimensions: M=1024, N=2058, K=4096")
    print("=" * 80)

    # Define the problem with backend-specific source requirements.
    implementation_kind = (
        "native MUSA extension bundle"
        if args.kernel_backend == "musa"
        else "fused Triton kernel"
    )
    problem_description = f"""
Write a correct, efficient {implementation_kind} for the following problem:

import torch
import torch.nn as nn

class Model(nn.Module):
def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.bfloat16))

    def forward(self, x):
        # Perform matmul and apply sigmoid activation
        output = torch.matmul(x, self.weight)
        output = torch.sigmoid(output)
        return output

# Define input dimensions and parameters
batch_size = 1024
in_features = 4096
out_features = 2058

def get_inputs():
    return [torch.randn(batch_size, in_features, dtype=torch.bfloat16)]

def get_init_inputs():
    return [in_features, out_features]
    """

    # Let the agent generate the test code
    print("\nGenerating kernel...")
    start_time = time.time()

    # Call agent to generate both test and kernel
    result = agent.generate_kernel(
        problem_description, test_code=None
    )  # Let agent generate test

    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f} seconds")

    # Print results
    if result["success"]:
        print("\n✓ Successfully generated BF16 matmul + sigmoid kernel!")
        print(
            f"  Worker {result['worker_id']} found solution in {result['rounds']} rounds"
        )
        print(f"  Session directory: {result['session_dir']}")

        print("\n" + "=" * 80)
        print("Generated Kernel Code:")
        print("=" * 80)
        print(result["kernel_code"])
        print("=" * 80)

        payload = result["kernel_code"]
        if isinstance(payload, dict):
            bundle_dir = os.path.join(result["session_dir"], "final_kernel")
            print(f"\n✓ Native kernel bundle saved to: {bundle_dir}")
            for name in sorted(payload):
                print(f"  - {name}")
        else:
            kernel_file = "bf16_matmul_sigmoid_kernel.py"
            with open(kernel_file, "w") as f:
                f.write(payload)
            print(f"\n✓ Kernel saved to: {kernel_file}")

        # Native MUSA bundles were already compiled and verified by the agent;
        # copying only kernel.py here would omit the compiled extension sources.
        if isinstance(payload, dict):
            agent.cleanup()
            print("\n✓ E2E test completed successfully!")
            return

        # Run the generated test to show performance
        print("\nRunning the generated test...")

        # Read the generated test code
        test_file = os.path.join(result["session_dir"], "test_0.py")
        with open(test_file, "r") as f:
            test_code = f.read()

        print("\nGenerated Test Code:")
        print("=" * 80)
        print(test_code)
        print("=" * 80)

        # Re-run the single-file Triton artifact for a visible final check.
        with open("kernel.py", "w") as f:
            f.write(payload)

        final_test_script = test_code

        with open("final_test.py", "w") as f:
            f.write(final_test_script)

        os.system("python final_test.py")

        # Cleanup kernel.py
        if os.path.exists("kernel.py"):
            os.remove("kernel.py")

        # Cleanup temporary test file
        os.remove("final_test.py")

    else:
        print("\n✗ Failed to generate kernel")
        print(f"  Message: {result['message']}")
        print(f"  Session directory: {result['session_dir']}")
        sys.exit(1)

    # Cleanup
    agent.cleanup()
    print("\n✓ E2E test completed successfully!")


if __name__ == "__main__":
    main()
