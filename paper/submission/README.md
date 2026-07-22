# arXiv submission materials

## `abstract.txt`

Paste into arXiv's **Abstract** metadata field. 1895 characters, against the field's 1920 limit.

The field is plain text: no TeX macros, no font commands, no math mode, no Unicode. This file is
pure ASCII with straight quotes, `--` for dashes, and `2.15x` / `p < 1e-4` / `O(|E|)` spelled out.
It is deliberately **not** the PDF's abstract, which is 3234 characters and keeps the full claim
set, the Spearman result, and the action-optimality figures. Both are accurate; this one is cut to
fit the form.

If you edit the paper's abstract, this file does not follow automatically. Re-check with:

```bash
python3 -c "t=open('paper/submission/abstract.txt').read().rstrip('\n'); print(len(t), 'chars')"
```

## Submitting

- **Primary category:** cs.AI. Reasonable cross-list: cs.NE (the PSO/GA ablation) or cs.SE.
- **Upload:** the `paper/` directory's `.tex`, `.bbl`, `refs.bib`, and the two `\input` files
  (`section_csp.tex`, `appendix_sweep.tex`) — all of which are on **this branch only**; main
  carries the code and `repro/`. arXiv does **not** run BibTeX, so `main.bbl` must be in the
  tarball — it is tracked in git for exactly this reason. `main.pdf` is untracked and must not be
  uploaded; arXiv builds from source and rejects a PDF bundled with TeX input. Exclude the hidden
  `.gitignore` too: arXiv rejects dotfiles.
- **Endorsement:** a first cs.AI submission may need one. Any author with cs.AI arXiv history can
  endorse.
- **Institutional email:** register/verify the DU address; arXiv gates submission on it.
- **Repository visibility:** ~~make the repository public~~ — done, the reproducibility URL
  resolves. (The `--accept-visibility-change-consequences` flag written here originally is not in
  gh 2.46, which is what this machine runs; there the flag is just `--visibility public`.)
