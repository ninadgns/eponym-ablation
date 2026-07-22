#!/usr/bin/env python
"""
Guard against companion-file drift.

`repro/README.md` is keyed to LaTeX \\label names rather than table numbers, because
numbers shift whenever a table is inserted and nothing in the build catches a stale
reference in a file LaTeX never reads. This script checks that:

  1. every `tab:...` / `fig:...` label cited in repro/README.md exists in paper/main.tex;
  2. every table and figure label defined in paper/main.tex is cited by repro/README.md;
  3. repro/README.md contains no bare "Table N" / "Figure N" references, which would drift.

The paper source lives on the `paper` branch, not on main. Off that branch there is
nothing to check against, so this exits 0 with a note rather than failing — a check
that cannot run is not a check that failed, and this is meant for CI.

Run from the repo root:  python repro/check_labels.py
Exits non-zero on any failure, so it can go in CI or a pre-commit hook.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "repro" / "README.md"
TEX = sorted((ROOT / "paper").glob("*.tex"))

if not TEX:
    print("SKIP — no paper/*.tex on this branch; the paper source lives on `paper`.")
    print("       To run the check:  git switch paper && python repro/check_labels.py")
    sys.exit(0)

tex_src = "\n".join(p.read_text() for p in TEX)
readme = README.read_text()

KINDS = ("tab", "fig")
_LABEL = r"(?:tab|fig):[a-zA-Z0-9-]+"

defined = set(re.findall(rf"\\label\{{({_LABEL})\}}", tex_src))
cited = set(re.findall(rf"`({_LABEL})`", readme))
bare = re.findall(r"^.*\b(?:Tables?|Figures?) \d+.*$", readme, flags=re.M)

failures = []

if missing := cited - defined:
    failures.append(f"README cites labels that do not exist in paper/: {sorted(missing)}")
if uncited := defined - cited:
    failures.append(f"paper/ defines labels the README never mentions: {sorted(uncited)}")
if bare:
    failures.append("README contains bare table/figure numbers, which drift:\n    "
                    + "\n    ".join(b.strip() for b in bare))

aux = ROOT / "paper" / "main.aux"
if aux.exists():
    nums = dict(re.findall(rf"\\newlabel\{{({_LABEL})\}}\{{\{{(\d+)\}}", aux.read_text()))
    print("current numbering (from main.aux):")
    for lab in sorted(nums, key=lambda k: (k.split(":")[0], int(nums[k]))):
        kind = "Table" if lab.startswith("tab:") else "Figure"
        print(f"  {kind} {nums[lab]:>2}  {lab}")
    print()

if failures:
    print("FAIL")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

counts = ", ".join(f"{sum(1 for d in defined if d.startswith(k + ':'))} {k}" for k in KINDS)
print(f"OK — {counts} labels, all defined in paper/ and all cited in repro/README.md")
