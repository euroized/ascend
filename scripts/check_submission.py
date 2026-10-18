"""Check a community benchmark submission and maintain the results board.

    # check a bench.json (prints every check and the verdict)
    python scripts/check_submission.py bench.json

    # accept it onto the board: stores results/community/<issue>-<submitter>.json, rebuilds the board
    python scripts/check_submission.py bench.json --issue 12 --submitter octocat --add

    # rebuild results/community/README.md from the stored submissions
    python scripts/check_submission.py --board

Criteria are those of docs/PREREG_PHASE1.md (Amendment 1); see ascend/submission.py.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend import data, submission  # noqa: E402

BOARD = data.REPO_ROOT / "results" / "community"
MAX_BYTES = 1 << 20


def mark(v) -> str:
    return {True: "✅", False: "❌", None: "—"}[v]


def load_records() -> list[dict]:
    recs = []
    for p in sorted(BOARD.glob("*.json")):
        r = json.loads(p.read_text())
        recs.append(
            {"id": p.stem, "issue": r["issue"], "submitter": r["submitter"], "verdict": submission.check(r["bench"])}
        )
    return recs


def render_board() -> str:
    recs = load_records()
    r4 = submission.rule4(recs)
    lines = [
        "# Community GPU results",
        "",
        "Every submission from the [Benchmark result](../../../../issues/new?template=benchmark_result.yml) "
        "issue template, checked by `scripts/check_submission.py` against "
        "[`docs/PREREG_PHASE1.md`](../../docs/PREREG_PHASE1.md) (Amendment 1). Generated file; do not edit by hand.",
        "",
        "| Submission | GPU | Driver | RTF (median) | Eligible | S1 ≥ 1.0 | F | G | Consumer | Rule 3 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        v = r["verdict"]
        lines.append(
            f"| [#{r['issue']}](../../../../issues/{r['issue']}) @{r['submitter']} | {v['gpu'] or '—'} | "
            f"{v['driver'] or '—'} | {v['rtf_median'] if v['rtf_median'] is not None else '—'} | "
            f"{mark(v['eligible'])} | {mark(v['S1'])} | {mark(v['F'])} | {mark(v['G'])} | "
            f"{mark(v['consumer_gpu'])} | {mark(v['rule3'])} |"
        )
    if not recs:
        lines.append("| *No submissions yet* | | | | | | | | | |")
    lines += [
        "",
        "**Real-time claim (rules 3 and 4):** "
        + (
            "met, by " + " and ".join(r4["pair"])
            if r4["met"]
            else "not met. It needs two rule-3 submissions from different people on different GPU models."
        ),
        "",
    ]
    return "\n".join(lines)


def write_board() -> Path:
    out = BOARD / "README.md"
    out.write_text(render_board())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", nargs="?", type=Path)
    ap.add_argument("--issue", type=int)
    ap.add_argument("--submitter", help="GitHub username of the person who ran it")
    ap.add_argument("--add", action="store_true", help="store the submission and rebuild the board")
    ap.add_argument("--board", action="store_true", help="only rebuild the board")
    args = ap.parse_args()

    BOARD.mkdir(parents=True, exist_ok=True)
    if args.board:
        print(f"wrote {write_board()}")
        return 0
    if args.bench is None:
        ap.error("give a bench.json, or --board")

    bench = json.loads(args.bench.read_text())
    v = submission.check(bench)
    for c in v["checks"]:
        print(
            f"[{ {True: 'PASS', False: 'FAIL', None: ' -- '}[c['pass']] }] {c['group']:9s} {c['check']}"
            + (f"  ({c['detail']})" if c["detail"] is not None else "")
        )
    print(f"GPU {v['gpu']}  RTF {v['rtf_median']}  eligible={v['eligible']}  rule3={v['rule3']}")

    if args.add:
        if args.issue is None or not args.submitter:
            ap.error("--add needs --issue and --submitter")
        rec = {
            "issue": args.issue,
            "submitter": args.submitter,
            "added_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bench": bench,
        }
        text = json.dumps(rec, indent=1) + "\n"
        if len(text.encode()) > MAX_BYTES:
            raise SystemExit("submission over 1 MB; not stored")
        dest = BOARD / f"{args.issue}-{args.submitter}.json"
        dest.write_text(text)
        print(f"stored {dest}; wrote {write_board()}")
    return 0 if v["rule3"] else 1


if __name__ == "__main__":
    sys.exit(main())
