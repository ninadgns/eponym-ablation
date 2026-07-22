# arXiv submission materials

## `abstract.txt`

Paste into arXiv's **Abstract** metadata field. 1847 characters, against the field's hard limit of
1920 — [arXiv rejects anything longer](https://info.arxiv.org/help/prep.html). The ~70 characters
of slack are deliberate: an earlier version sat at 1895, which is within the limit but close enough
that any edit to the paper's abstract could push this past it without an obvious signal.

The field is plain text: no TeX macros, no font commands, no math mode, no Unicode. This file is
pure ASCII with straight quotes, `--` for dashes, and `2.15x` / `p < 1e-4` / `O(|E|)` spelled out.
It is deliberately **not** the PDF's abstract, which is 2796 characters of TeX and keeps the
$\rho = -1.0000$ result, the 10/10 invariance check, the CSP significance levels, and the
action-optimality figures. Both are accurate; this one is cut to fit the form.

If you edit the paper's abstract, this file does not follow automatically. Re-check with:

```bash
python3 -c "t=open('paper/submission/abstract.txt').read().rstrip('\n'); print(len(t), 'chars')"
```

arXiv also rejects Unicode in this field and strips carriage returns not followed by leading
whitespace, so keep it ASCII and keep the paragraph breaks as bare blank lines. To verify both:

```bash
python3 -c "
t=open('paper/submission/abstract.txt').read().rstrip('\n')
print(len(t),'chars', 'OK' if len(t)<=1920 else 'TOO LONG')
print('non-ascii:', [c for c in t if ord(c)>127] or 'none')"
```

## Submitting

- **Primary category:** cs.AI. Reasonable cross-list: cs.NE (the PSO/GA ablation) or cs.SE.
- **Upload:** the `paper/` directory's `.tex`, `.bbl`, `refs.bib`, the two `\input` files
  (`section_csp.tex`, `appendix_sweep.tex`), and `figures/weff.pdf` + `figures/pso_ablation.pdf`
  — all of which are on **this branch only**; main carries the code and `repro/`. arXiv does
  **not** run BibTeX, so `main.bbl` must be in the tarball — it is tracked in git for exactly this
  reason. Keep the figures in a `figures/` subdirectory; `\graphicspath{{figures/}}` in the
  preamble means the `\includegraphics` calls carry no path and resolve either way. The figures
  are vector PDFs, which pdflatex takes directly. `main.pdf` is untracked and must not be
  uploaded; arXiv builds from source and rejects a PDF bundled with TeX input, and a figure named
  `main.pdf` would collide besides. Exclude the hidden `.gitignore` too: arXiv rejects dotfiles.

  ```bash
  cd paper && tar czf ../submission.tar.gz \
    main.tex section_csp.tex appendix_sweep.tex main.bbl refs.bib figures/*.pdf
  ```
- **Endorsement:** a first cs.AI submission may need one. Any author with cs.AI arXiv history can
  endorse.
- **Institutional email:** register/verify the DU address; arXiv gates submission on it.
- **Repository visibility:** ~~make the repository public~~ — done, the reproducibility URL
  resolves. (The `--accept-visibility-change-consequences` flag written here originally is not in
  gh 2.46, which is what this machine runs; there the flag is just `--visibility public`.)
