"""P0 vs P3: did the core merge move any number?

Two tiers, because the four rows are not four samples of the same kind of quantity
(``../21-core-merge.md`` §9.2):

* **exact** — ade20k mIoU and in1k top1/top5 are integer-derived (confusion counts,
  correct/N), so on one machine they move only when a *prediction* flips. Any movement at
  all is a real change.
* **1e-5** — distill's scalars are float means of cosine similarities with nothing
  quantizing them, so they carry the hardware offset F1 describes. Demanding bit-identity
  there would fail the gate for reasons unrelated to the merge.

Run: ``.venv-cu126/bin/python unification_docs/phase2_baseline/compare.py``
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
TOL = 1e-5
ROWS = [
    ("ade20k_fixgrid.json", "exact"),
    ("ade20k_full_pin2.json", "exact"),
    ("in1k_fixgrid.json", "exact"),
    ("distill.json", "tol"),
]


def flatten(m: object, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(m, dict):
        for k, v in m.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(m, (int, float)) and not isinstance(m, bool):
        out[prefix[:-1]] = m
    return out


def main() -> int:
    total = violations = 0
    worst = 0.0
    for name, rule in ROWS:
        before = json.loads((HERE / f"p0_{name}").read_text())
        after = json.loads((HERE / f"p3_{name}").read_text())
        fb, fa = flatten(before["metrics"]), flatten(after["metrics"])
        assert set(fb) == set(fa), f"{name}: metric keys differ: {set(fb) ^ set(fa)}"
        bad = []
        row_worst = 0.0
        for k in sorted(fb):
            d = abs(fb[k] - fa[k])
            row_worst = max(row_worst, d)
            if (rule == "exact" and d != 0.0) or (rule == "tol" and d > TOL):
                bad.append((k, fb[k], fa[k], d))
        # A packaging change must not alter the resolved protocol either.
        if before.get("protocol") != after.get("protocol"):
            bad.append(("<protocol>", before.get("protocol"), after.get("protocol"), float("nan")))
        total += len(fb)
        violations += len(bad)
        worst = max(worst, row_worst)
        verdict = "PASS" if not bad else "FAIL"
        shown = "EXACT" if row_worst == 0.0 else f"max |diff| {row_worst:.3e}"
        print(f"{name:24s} rule={rule:5s} {len(fb):2d} scalars  {shown}  -> {verdict}")
        for k, x, y, d in bad[:5]:
            print(f"    {k}: {x!r} vs {y!r}  ({d:.3e})")
    print(f"\n{total} scalars compared, {violations} violations, worst |diff| = {worst:.3e}")
    print("P3:", "PASS" if violations == 0 else "FAIL")
    return 0 if violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
