#!/usr/bin/env python
"""
Guard against companion-file drift.

`repro/README.md` is keyed to LaTeX \\label names rather than table numbers, because
numbers shift whenever a table is inserted and nothing in the build catches a stale
reference in a file LaTeX never reads. This script checks that:

  1. every `tab:...` label cited in repro/README.md exists in paper/main.tex;
  2. every table label defined in paper/main.tex is cited by repro/README.md;
  3. repro/README.md contains no bare "Table N" references, which would drift.

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

tex_src = "\n".join(p.read_text() for p in TEX)
readme = README.read_text()

defined = set(re.findall(r"\\label\{(tab:[a-zA-Z0-9-]+)\}", tex_src))
cited = set(re.findall(r"`(tab:[a-zA-Z0-9-]+)`", readme))
bare = re.findall(r"^.*\bTables? \d+.*$", readme, flags=re.M)

failures = []

if missing := cited - defined:
    failures.append(f"README cites labels that do not exist in paper/: {sorted(missing)}")
if uncited := defined - cited:
    failures.append(f"paper/ defines table labels the README never mentions: {sorted(uncited)}")
if bare:
    failures.append("README contains bare table numbers, which drift:\n    "
                    + "\n    ".join(b.strip() for b in bare))

aux = ROOT / "paper" / "main.aux"
if aux.exists():
    nums = dict(re.findall(r"\\newlabel\{(tab:[a-zA-Z0-9-]+)\}\{\{(\d+)\}", aux.read_text()))
    print("current numbering (from main.aux):")
    for lab in sorted(nums, key=lambda k: int(nums[k])):
        print(f"  Table {nums[lab]:>2}  {lab}")
    print()

if failures:
    print("FAIL")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print(f"OK — {len(defined)} table labels, all defined in paper/ and all cited in repro/README.md")
