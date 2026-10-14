"""Inspect and populate evidence-backed KernelAgent experiments.

Run with ``python -m scripts.experience_db --help`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from triton_kernel_agent.experience import SQLiteExperienceStore
from triton_kernel_agent.experience import build_operator_signature
from triton_kernel_agent.experience.experiment_db import (
    lesson_report,
    record_bundle,
    run_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=Path.cwd() / ".kernelagent/experiences.sqlite3",
        help="SQLite experience database (default: .kernelagent/experiences.sqlite3)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="Record one JSON experiment bundle")
    record.add_argument("input", type=Path)
    show = commands.add_parser("show", help="Show a run with its measurements")
    show.add_argument("run_id")
    lessons = commands.add_parser("lessons", help="Show lessons and evidence counts")
    lessons.add_argument("--limit", type=int, default=20)
    lessons.add_argument("--semantic-type", default="")
    lessons.add_argument("--platform", default="")
    lessons.add_argument("--kernel-backend", default="")
    retrieve = commands.add_parser(
        "retrieve", help="Preview experience selected for a problem file"
    )
    retrieve.add_argument("problem", type=Path)
    retrieve.add_argument("--platform", required=True)
    retrieve.add_argument("--kernel-backend", required=True)
    retrieve.add_argument("--stage", choices=("generation", "optimization"),
                          default="generation")
    retrieve.add_argument("--limit", type=int, default=4)
    retrieve.add_argument("--direction", choices=("forward", "backward", "joint", "unknown"))
    retrieve.add_argument("--hardware", default="")
    retrieve.add_argument("--architecture", default="")
    retrieve.add_argument("--layout", default="")
    retrieve.add_argument("--trace-graph", action="store_true",
                          help="Probe the reference Model for structural text recall")
    retrievals = commands.add_parser(
        "retrievals", help="Inspect which experience IDs were shown to sessions"
    )
    retrievals.add_argument("--session-id", default="")
    retrievals.add_argument("--limit", type=int, default=20)
    commands.add_parser("stats", help="Count workloads, runs, attempts, cases, lessons, retrievals")
    commands.add_parser("merge-proposals", help="Suggest compatible claim duplicates; never auto-merge")
    merge = commands.add_parser("merge-claims", help="Explicitly approve a reversible claim alias")
    merge.add_argument("source_id"); merge.add_argument("target_id")
    merge.add_argument("--reviewer", required=True); merge.add_argument("--reason", required=True)
    undo = commands.add_parser("unmerge-claim", help="Remove an alias without altering raw evidence")
    undo.add_argument("source_id")
    undo.add_argument("--reviewer",required=True); undo.add_argument("--reason",required=True)
    args = parser.parse_args(argv)

    store = SQLiteExperienceStore(args.db)
    with store._connect() as connection:
        if args.command == "merge-proposals":
            from triton_kernel_agent.experience.consolidation import proposals
            output = proposals(connection)
        elif args.command == "merge-claims":
            from triton_kernel_agent.experience.consolidation import review_merge
            review_merge(connection,args.source_id,args.target_id,reviewer=args.reviewer,reason=args.reason)
            output = {"aliased":args.source_id,"canonical":args.target_id}
        elif args.command == "unmerge-claim":
            from triton_kernel_agent.experience.consolidation import review_unmerge
            changed=review_unmerge(connection,args.source_id,reviewer=args.reviewer,reason=args.reason)
            output = {"source_id":args.source_id,"alias_revoked":changed,"raw_evidence_unchanged":True}
        elif args.command == "record":
            bundle = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(bundle, dict):
                raise ValueError("input must be a JSON object")
            output = {"run_id": record_bundle(connection, bundle)}
        elif args.command == "show":
            output = run_report(connection, args.run_id)
            if output is None:
                parser.error(f"run not found: {args.run_id}")
        elif args.command == "lessons":
            output = lesson_report(
                connection, max(0, args.limit),
                semantic_type=args.semantic_type, platform=args.platform,
                kernel_backend=args.kernel_backend,
            )
        elif args.command == "retrieve":
            signature = build_operator_signature(
                args.problem.read_text(encoding="utf-8"), trace_graph=args.trace_graph,
            )
            selected = store.retrieve(
                signature, args.platform, args.kernel_backend,
                stage=args.stage, limit=args.limit,
                direction=args.direction, hardware=args.hardware, layout=args.layout,
                architecture=args.architecture,
            )
            output = {
                "signature": signature.to_dict(),
                "selected": [{
                    "id": item.id, "kind": item.kind,
                    "outcome": item.outcome, "scope": item.scope,
                    "retrieval_match": item.retrieval_match,
                    "retrieval_reason": item.retrieval_reason,
                    "text_evidence_status": item.text_evidence_status if item.kind == "case" else "validated_principle",
                    "graph_match_evidence": item.graph_match_evidence,
                    "claim_conditions": item.claim_conditions,
                    "applicability": item.environment.get("principle_applicability", {}),
                    "lesson": item.lesson, "action": item.action,
                } for item in selected],
                "seed_ids": [item.id for item in store.search_verified_seeds(
                    signature, args.platform, args.kernel_backend,
                    direction=args.direction, hardware=args.hardware, layout=args.layout,
                )],
            }
        elif args.command == "retrievals":
            if args.session_id:
                rows = connection.execute(
                    """SELECT * FROM experience_retrieval_events
                    WHERE session_id = ? ORDER BY created_at DESC LIMIT ?""",
                    (args.session_id, max(0, args.limit)),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM experience_retrieval_events
                    ORDER BY created_at DESC LIMIT ?""",
                    (max(0, args.limit),),
                ).fetchall()
            output = [{
                "id": row["id"], "session_id": row["session_id"],
                "stage": row["stage"],
                "case_ids": json.loads(row["case_ids_json"]),
                "seed_ids": json.loads(row["seed_ids_json"]),
                "selection": json.loads(row["selection_json"]),
                "outcome": json.loads(row["outcome_json"]),
                "created_at": row["created_at"],
            } for row in rows]
        else:
            tables = {
                "workloads": "experiment_workloads",
                "runs": "experiment_runs",
                "attempts": "experiment_attempts",
                "measurements": "experiment_measurements",
                "claims": "experiment_claims",
                "claim_evidence": "experiment_claim_evidence",
                "retrievals": "experience_retrieval_events",
            }
            output = {
                name: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for name, table in tables.items()
            }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
