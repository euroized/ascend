"""Evaluate the sugar GRN -> MN9 reproduction criteria of docs/PREREG_PHASE0.md.

Reads results/phase0/reference_rates.json (from scripts/reference_rates.py), prints every check
and writes results/phase0/sugar_mn9.json. Exit code 0 only if all criteria pass.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend import data  # noqa: E402

RATES = data.REPO_ROOT / "results" / "phase0" / "reference_rates.json"
OUT = data.REPO_ROOT / "results" / "phase0" / "sugar_mn9.json"

# Shiu et al.'s published model output (results/example/sugarR*.parquet in philshiu/Drosophila_brain_model
# @ 91bdd1e, FlyWire v630, 30 trials): MN9 mean rate in Hz at each sugar GRN stimulation rate.
PUBLISHED_V630 = {200: {"mn9_left": 93.2667, "mn9_right": 61.8000},
                  100: {"mn9_left": 67.0333, "mn9_right": 48.6333}}
R1_TOL_HZ = 5.0        # harness validation on v630: |ours - published| <= 5 Hz
R2_MIN_HZ = 10.0       # v783 sugar at 150 Hz drives each MN9 to >= 10 Hz
R2_BITTER_MAX_HZ = 1.0 # v783 bitter alone keeps each MN9 below 1 Hz
R3_REL_TOL = 0.25      # v783 MN9 rates within +-25 % of the published v630 rates
MN9 = ("mn9_left", "mn9_right")


def main() -> int:
    c = json.loads(RATES.read_text())["conditions"]
    rate = (lambda cond, k: c[cond]["readouts"][k]["rate_hz_mean"])
    checks = []

    def check(rule: str, desc: str, value, ok: bool):
        checks.append({"rule": rule, "check": desc, "value": value, "pass": bool(ok)})

    for hz, pub in PUBLISHED_V630.items():
        for k in MN9:
            ours = rate(f"v630_sugar_{hz}hz", k)
            check("R1", f"v630 sugar {hz} Hz: {k} {ours:.2f} vs published {pub[k]:.2f} (tol {R1_TOL_HZ} Hz)",
                  round(ours - pub[k], 3), abs(ours - pub[k]) <= R1_TOL_HZ)

    for k in MN9:
        ours = rate("v783_sugar_150hz", k)
        check("R2", f"v783 sugar 150 Hz: {k} {ours:.2f} Hz >= {R2_MIN_HZ}", ours, ours >= R2_MIN_HZ)
    none = c["v783_none"]
    check("R2", "v783 no stimulus: zero spikes in every trial", max(none["spikes_per_trial"]),
          max(none["spikes_per_trial"]) == 0)
    for k in MN9:
        ours = rate("v783_bitter_150hz", k)
        check("R2", f"v783 bitter 150 Hz: {k} {ours:.2f} Hz < {R2_BITTER_MAX_HZ}", ours, ours < R2_BITTER_MAX_HZ)

    for hz, pub in PUBLISHED_V630.items():
        for k in MN9:
            ours = rate(f"v783_sugar_{hz}hz", k)
            rel = (ours - pub[k]) / pub[k]
            check("R3", f"v783 sugar {hz} Hz: {k} {ours:.2f} vs published v630 {pub[k]:.2f} ({rel:+.1%}, tol "
                  f"+-{R3_REL_TOL:.0%})", round(rel, 4), abs(rel) <= R3_REL_TOL)

    r = [rate(f"v783_sugar_{hz}hz", "mn9_left") for hz in (100, 150, 200)]
    check("R4", f"v783 mn9_left rises with sugar rate: {r[0]:.2f} < {r[1]:.2f} < {r[2]:.2f}", r, r[0] < r[1] < r[2])

    rules = {rule: all(x["pass"] for x in checks if x["rule"] == rule) for rule in ("R1", "R2", "R3", "R4")}
    verdict = "REPRODUCED" if all(rules.values()) else "NOT REPRODUCED"
    for x in checks:
        print(f"[{'PASS' if x['pass'] else 'FAIL'}] {x['rule']} {x['check']}")
    print(f"rules: {rules}\nverdict: {verdict}")
    OUT.write_text(json.dumps({"prereg": "docs/PREREG_PHASE0.md", "rules": rules, "verdict": verdict,
                               "checks": checks}, indent=1) + "\n")
    return 0 if verdict == "REPRODUCED" else 1


if __name__ == "__main__":
    sys.exit(main())
