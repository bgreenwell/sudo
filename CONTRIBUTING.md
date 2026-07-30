# Contributing

Thanks for taking a look. This repository backs a methods paper, so most
contributions are either a methodological change or a fix to the code or
manuscript.

## Workflow: R first, then Python

Every methodological change is prototyped and validated in an `R/stageN_*.R`
script with explicit acceptance criteria (bias against Monte-Carlo standard
error, coverage bands) before it is ported to the Python package. Do not add
method code to Python without a validated R stage behind it. See `AGENTS.md`
for the design decisions and gotchas.

## Running things

```bash
Rscript R/stage0_fwl.R            # any stage; run from the repository root
cd python && uv sync && uv run pytest
cd manuscript/paper && quarto render sudo_paper.qmd --to arxiv-pdf
```

## Conventions

- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`,
  `chore:`, `ci:`), imperative mood, first line under 72 characters.
- No em dashes or emoji in code, comments, commit messages, or docs.
- Branch off `main`; PRs are merged, not squashed.

## Feedback on the paper

For comments on the manuscript or method, open an issue using the feedback
template.
