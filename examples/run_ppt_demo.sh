#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROBLEM_FILE="$SCRIPT_DIR/optimize_04_musa_sigmoid/problem.py"
OUTPUT_ROOT="$SCRIPT_DIR/results"
ALLOW_API=0
REASONING_EFFORT=""
KERNEL_SEEDS=""
GENERATION_ROUNDS=""
OPTIMIZATION_ROUNDS=""
REUSE_GENERATED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-third-party-api) ALLOW_API=1 ;;
        --keep-artifacts) : ;; # Compatibility alias; results persist by default.
        --problem|--output-root|--kernel-seeds|--generation-rounds|--optimization-rounds|--reuse-generated)
            option="$1"
            shift
            [[ $# -gt 0 ]] || { echo "$option needs a value" >&2; exit 2; }
            case "$option" in
                --problem) PROBLEM_FILE="$1" ;;
                --output-root) OUTPUT_ROOT="$1" ;;
                --kernel-seeds) KERNEL_SEEDS="$1" ;;
                --generation-rounds) GENERATION_ROUNDS="$1" ;;
                --optimization-rounds) OPTIMIZATION_ROUNDS="$1" ;;
                --reuse-generated) REUSE_GENERATED="$1" ;;
            esac
            ;;
        --reasoning-effort)
            shift
            [[ $# -gt 0 ]] || { echo '--reasoning-effort needs a value' >&2; exit 2; }
            REASONING_EFFORT="$1"
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash examples/run_ppt_demo.sh [options]

Options:
  --problem FILE          KernelBench-style problem (default: MUSA sigmoid example).
  --output-root DIR       Persistent run root (default: examples/results).
  --allow-third-party-api  Required for LLM AutoRoute, generation and optimization.
  --reasoning-effort NAME Select none/low/medium/high/xhigh/max; otherwise use .env.
  --kernel-seeds N        Generation workers/seeds; otherwise use .env (default 1).
  --generation-rounds N   Maximum generation/refinement rounds (default 10).
  --optimization-rounds N Maximum MCU-guided optimization rounds (default 1).
  --reuse-generated DIR   Skip generation and reuse a verified prior result directory.

Presentation command:
  bash examples/run_ppt_demo.sh \
    --problem examples/optimize_04_musa_sigmoid/problem.py \
    --allow-third-party-api --reasoning-effort low
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

cd "$REPO_ROOT"
export LD_LIBRARY_PATH="/usr/local/musa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PROBLEM_FILE="$(readlink -f -- "$PROBLEM_FILE")"
OUTPUT_ROOT="$(readlink -m -- "$OUTPUT_ROOT")"
[[ -f "$PROBLEM_FILE" ]] || { echo "Problem file not found: $PROBLEM_FILE" >&2; exit 2; }
mapfile -t ENV_DEFAULTS < <(python3 - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path.cwd() / ".env")
print(os.getenv("NUM_KERNEL_SEEDS", "1"))
print(os.getenv("MAX_REFINEMENT_ROUNDS", "10"))
PY
)
KERNEL_SEEDS="${KERNEL_SEEDS:-${ENV_DEFAULTS[0]}}"
GENERATION_ROUNDS="${GENERATION_ROUNDS:-${ENV_DEFAULTS[1]}}"
OPTIMIZATION_ROUNDS="${OPTIMIZATION_ROUNDS:-1}"
for value in "$KERNEL_SEEDS" "$GENERATION_ROUNDS" "$OPTIMIZATION_ROUNDS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo 'Worker and round counts must be positive integers' >&2
        exit 2
    }
done
export KERNEL_SEEDS GENERATION_ROUNDS OPTIMIZATION_ROUNDS
if [[ -n "$REASONING_EFFORT" ]]; then
    case "$REASONING_EFFORT" in
        none|low|medium|high|xhigh|max) ;;
        *) echo 'Reasoning effort must be none/low/medium/high/xhigh/max' >&2; exit 2 ;;
    esac
    export OPENAI_REASONING_EFFORT="$REASONING_EFFORT"
fi

RUN_ID="run_$(date +%Y%m%d_%H%M%S)_$$"
DEMO_WORKDIR="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$DEMO_WORKDIR"
exec > >(tee -a "$DEMO_WORKDIR/demo.log") 2>&1
export DEMO_WORKDIR PROBLEM_FILE
cleanup() {
    exit_code=$?
    trap - EXIT
    echo "Runtime artifacts saved at: $DEMO_WORKDIR"
    exit "$exit_code"
}
trap cleanup EXIT

if [[ -n "$REUSE_GENERATED" ]]; then
    REUSE_GENERATED="$(readlink -f -- "$REUSE_GENERATED")"
    for name in kernel.py binding.cpp kernel.mu setup.py problem.py test.py; do
        [[ -f "$REUSE_GENERATED/$name" ]] || {
            echo "Reuse directory is missing $name: $REUSE_GENERATED" >&2
            exit 2
        }
        cp -- "$REUSE_GENERATED/$name" "$DEMO_WORKDIR/$name"
    done
    PROBLEM_FILE="$DEMO_WORKDIR/problem.py"
    export PROBLEM_FILE REUSE_GENERATED
fi

section() {
    printf '\n%s\n%s\n%s\n' \
        '================================================================================' "$1" \
        '================================================================================'
}

section '1. MUSA / MCU / model preflight'
python3 - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")
import torch
import torch_musa  # noqa: F401

x = torch.ones(1, device=torch.device("musa"))
print(f"MUSA available : {torch.musa.is_available()}")
print(f"Device         : {x.device}")
print(f"Model          : {os.getenv('OPENAI_MODEL', '(not set)')}")
print(f"Reasoning      : {os.getenv('OPENAI_REASONING_EFFORT', '(provider default)')}")
print(f"API key loaded : {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"Kernel seeds   : {os.environ['KERNEL_SEEDS']}")
print(f"Generate rounds: {os.environ['GENERATION_ROUNDS']}")
print(f"Optimize rounds: {os.environ['OPTIMIZATION_ROUNDS']}")
PY
mcu --version

section '2. AutoRoute static evidence'
python3 - <<'PY'
import os
from pathlib import Path
from Fuser.auto_agent import analyze_problem_code

problem = Path(os.environ["PROBLEM_FILE"])
analysis = analyze_problem_code(problem.read_text(encoding="utf-8"))
route = "fuser" if analysis.route_to_fuser() else "kernelagent"
print(f"Problem       : {problem}")
print(f"Static route  : {route}")
print(f"Complexity    : {analysis}")
PY

if [[ "$ALLOW_API" -ne 1 ]]; then
    section 'Online stages skipped safely'
    echo 'Preflight and static AutoRoute passed.'
    echo 'Rerun with --allow-third-party-api for LLM routing and the full pipeline.'
    exit 0
fi

if [[ -z "$REUSE_GENERATED" ]]; then
    section '3. LLM AutoRoute -> generate -> compile -> numerical verification'
    python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from Fuser.auto_agent import AutoKernelRouter

load_dotenv(Path.cwd() / ".env")
problem = Path(os.environ["PROBLEM_FILE"]).resolve()
runtime = Path(os.environ["DEMO_WORKDIR"])
model = os.getenv("OPENAI_MODEL", "deepseek-chat")
router = AutoKernelRouter(
    ka_model=model,
    ka_num_workers=int(os.environ["KERNEL_SEEDS"]),
    ka_max_rounds=int(os.environ["GENERATION_ROUNDS"]),
    ka_high_reasoning=False,
    router_model=model,
    router_high_reasoning=False,
    extract_model=model,
    dispatch_model=model,
    compose_model=model,
    target_platform="musa",
    kernel_backend="musa",
    allow_fallback=True,
    use_router_cache=False,
    verify=True,
    run_timeout_s=600,
    test_timeout_s=600,
    ka_log_dir=str(runtime / "generation_logs"),
)
result = router.solve(problem)
route_record = {"route": result.route, "success": result.success, "details": result.details}
print(json.dumps(route_record, indent=2, default=str))
(runtime / "route_result.json").write_text(
    json.dumps(route_record, indent=2, default=str), encoding="utf-8"
)
if not result.success:
    raise SystemExit("AutoRoute pipeline failed")
shutil.copy2(problem, runtime / "problem.py")
if result.route == "fuser":
    if isinstance(result.kernel_code, str):
        (runtime / "composed_kernel.py").write_text(
            result.kernel_code, encoding="utf-8"
        )
    print(f"Fuser artifacts: {result.details.get('run_dir', '(see route_result.json)')}")
    print("Fuser route verified; single-Kernel MCU optimization is not applicable.")
    raise SystemExit(0)
if result.route != "kernelagent" or not isinstance(result.kernel_code, dict):
    raise SystemExit(f"Unsupported AutoRoute result: {result.route}")

required = ("kernel.py", "binding.cpp", "kernel.mu", "setup.py")
for name in required:
    content = result.kernel_code.get(name)
    if not content:
        raise SystemExit(f"Generated bundle is missing {name}")
    (runtime / name).write_text(content, encoding="utf-8")
session = Path(result.details["session_dir"])
test_candidates = [session / "test_0.py", session / "test.py"]
test_source = next((path for path in test_candidates if path.is_file()), None)
if test_source is None:
    raise SystemExit(f"Generated correctness test not found in {session}")
shutil.copy2(test_source, runtime / "test.py")
print(f"Generated runtime bundle: {runtime}")
print("Bundle files:", ", ".join(required))
PY
else
    section '3. Reuse verified generated Kernel bundle'
    python3 - <<'PY'
import json
import os
from pathlib import Path

runtime = Path(os.environ["DEMO_WORKDIR"])
record = {
    "route": "kernelagent",
    "success": True,
    "details": {"reused_from": os.environ["REUSE_GENERATED"]},
}
(runtime / "route_result.json").write_text(
    json.dumps(record, indent=2), encoding="utf-8"
)
print(f"Reused verified bundle from: {os.environ['REUSE_GENERATED']}")
print(f"Copied into new run: {runtime}")
PY
fi

DEMO_ROUTE="$(python3 -c 'import json,os; print(json.load(open(os.path.join(os.environ["DEMO_WORKDIR"], "route_result.json")))["route"])')"
if [[ "$DEMO_ROUTE" == "fuser" ]]; then
    section '4. Fuser pipeline result'
    python3 - <<'PY'
import json
import os
from pathlib import Path

runtime = Path(os.environ["DEMO_WORKDIR"])
route = json.loads((runtime / "route_result.json").read_text(encoding="utf-8"))
summary = {
    "status": "SUCCESS",
    "route": "fuser",
    "verified": True,
    "optimization_status": "not_applicable_to_composed_fuser_output",
    "result_dir": str(runtime),
    "fuser_run_dir": route.get("details", {}).get("run_dir"),
}
(runtime / "demo_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
    section 'PPT demo completed: Fuser pipeline generated and verified the composed result'
    exit 0
fi

section '4. MCU profile -> bottleneck -> GPT optimization -> reverify -> benchmark'
RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
export RUN_STARTED_AT
python3 examples/run_opt_manager.py \
    --strategy musa \
    --kernel-dir "$DEMO_WORKDIR" \
    --max-rounds "$OPTIMIZATION_ROUNDS" \
    --result-json "$DEMO_WORKDIR/optimization_result.json"

section '5. Run-specific result, best bundle and persisted experience'
python3 - <<'PY'
import json
import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from triton_kernel_agent.experience import build_operator_signature

load_dotenv(Path.cwd() / ".env")
runtime = Path(os.environ["DEMO_WORKDIR"])
optimization = json.loads(
    (runtime / "optimization_result.json").read_text(encoding="utf-8")
)
route = json.loads((runtime / "route_result.json").read_text(encoding="utf-8"))

metric_files = sorted(runtime.glob("opt_manager_logs/**/*_mcu_metrics.json"))
log_text = "\n".join(
    path.read_text(encoding="utf-8", errors="replace")
    for path in runtime.glob("opt_manager_logs/**/*.log")
)
if metric_files:
    mcu_source = "live"
elif "Using source-hash-matched MCU profile cache" in log_text:
    mcu_source = "source_hash_cache"
else:
    mcu_source = "unavailable"

bottleneck = None
sol_pct = None
matches = re.findall(
    r"(?:Baseline profiled.*?:|Baseline SOL:)\s*(?:[0-9.]+%\s*\()?"
    r"([a-z_-]+)-bound.*?([0-9.]+)% SOL",
    log_text,
)
if matches:
    bottleneck, sol_text = matches[-1]
    sol_pct = float(sol_text)

experience_db = Path(".kernelagent/experiences.sqlite3")
row = None
verified_seed_count = 0
problem_code = (runtime / "problem.py").read_text(encoding="utf-8")
signature = build_operator_signature(problem_code)
if experience_db.is_file():
    connection = sqlite3.connect(f"file:{experience_db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        """SELECT id, semantic_type, outcome, verified, action, bottleneck,
                  baseline_time_ms, kernel_time_ms, improvement_pct, created_at
           FROM experiences
           WHERE platform='musa' AND kernel_backend='musa'
             AND signature_digest=? AND datetime(created_at) >= datetime(?)
           ORDER BY created_at DESC LIMIT 1""",
        (signature.digest, os.environ["RUN_STARTED_AT"]),
    ).fetchone()
    verified_seed_count = connection.execute(
        """SELECT COUNT(*) FROM experiences
           WHERE platform='musa' AND kernel_backend='musa'
             AND signature_digest=? AND verified=1
             AND outcome IN ('success', 'improved')""",
        (signature.digest,),
    ).fetchone()[0]

loaded_matches = re.findall(r"Loaded (\d+) relevant cross-task experiences", log_text)
experiences_loaded = int(loaded_matches[-1]) if loaded_matches else 0

experience = dict(row) if row is not None else None
summary = {
    "status": optimization["status"],
    "route": route["route"],
    "model": os.getenv("OPENAI_MODEL", "deepseek-chat"),
    "reasoning_effort": os.getenv("OPENAI_REASONING_EFFORT"),
    "kernel_seeds": int(os.environ["KERNEL_SEEDS"]),
    "generation_rounds_limit": int(os.environ["GENERATION_ROUNDS"]),
    "optimization_rounds_limit": int(os.environ["OPTIMIZATION_ROUNDS"]),
    "generation_reused": bool(os.getenv("REUSE_GENERATED")),
    "mcu_source": mcu_source,
    "bottleneck": bottleneck,
    "sol_pct": sol_pct,
    **optimization,
    "experience_saved_this_run": experience is not None,
    "experience": experience,
    "cross_task_experiences_loaded": experiences_loaded,
    "verified_seed_count": verified_seed_count,
    "best_bundle_dir": optimization.get("best_bundle_dir"),
    "result_dir": str(runtime),
}
(runtime / "demo_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)

print(f"Status          : {summary['status']}")
print(f"Route           : {summary['route']}")
print(f"MCU source      : {summary['mcu_source']}")
print(f"Initial time    : {summary['initial_time_ms']} ms")
print(f"Best time       : {summary['best_time_ms']} ms")
print(f"Improvement     : {summary['improvement_pct']} %")
print(f"Best bundle     : {summary['best_bundle_dir']}")
print(f"Experience saved: {summary['experience_saved_this_run']}")
print(f"Experience loaded: {summary['cross_task_experiences_loaded']}")
print(f"Verified seeds   : {summary['verified_seed_count']}")
print(f"Summary JSON    : {runtime / 'demo_summary.json'}")
PY

DEMO_STATUS="$(python3 -c 'import json,os; print(json.load(open(os.path.join(os.environ["DEMO_WORKDIR"], "demo_summary.json")))["status"])')"
case "$DEMO_STATUS" in
    SUCCESS)
        section 'PPT demo completed: optimized Kernel improved'
        ;;
    NO_GAIN)
        section 'PPT demo completed: valid candidate, initial Kernel remains best'
        ;;
    DEGRADED)
        section 'PPT demo degraded: no successful optimization candidate'
        echo "Inspect MCU and worker logs under: $DEMO_WORKDIR/opt_manager_logs" >&2
        exit 3
        ;;
    *)
        section 'PPT demo failed'
        exit 1
        ;;
esac
echo "Generated and optimized results: $DEMO_WORKDIR"
