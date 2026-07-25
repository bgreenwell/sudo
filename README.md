# SUDO — Surrogate-Assisted Double Machine Learning

Binary and ordinal outcomes have no home in the DML partially linear model:
the additive structure lives on a latent scale, and existing tooling covers
only the binary-logistic case. SUDO completes the discrete outcome to a
continuous latent-utility surrogate (a truncated inverse-transform draw from
the assumed link law), runs standard FWL partialling-out on each completed
dataset, and pools B proper-MI draws with Rubin's rules — restoring the
Neyman-orthogonal linear score, valid coverage, and link flexibility
(logit, cloglog) for binary and ordinal outcomes alike.

Development is R-first: each methodological step is a standalone script in
`R/` with explicit acceptance criteria, validated before being ported to the
Python package.

## Validation ladder

```bash
Rscript R/stage0_fwl.R                 # FWL exactness; cross-fit PLR sanity
Rscript R/stage1_binary_surrogate.R    # binary surrogate FWL, single draw
Rscript R/stage2_binary_rubin.R        # Rubin pooling: naive vs improper vs proper MI
Rscript R/stage3_binary_dml.R          # full binary SUDO with ML nuisances
Rscript R/stage4_ordinal_simple.R      # ordinal J=3, clm full model
Rscript R/stage5_ordinal_dml.R         # ordinal with flexible nuisances
Rscript R/stage6_link_misspec.R        # wrong-link bias + variance-drift diagnostic
```

Each script prints its Monte Carlo table, asserts its pass criteria, and
writes a summary CSV to `R/results/`.

## Python package

```bash
cd python && uv sync && uv run pytest
```

## Structure

```
R/            validation ladder + sourced helpers (R/sudo/)
python/       package (src/sudo) and tests
manuscript/   theory notes (sudo.md), Quarto paper, literature
archive/      frozen v1 experiments
```

## Key references

- Chernozhukov et al. (2018) — Double/debiased machine learning
- Liu, Zhang & Zhou (2021) — Logistic PLR DML (arXiv 2009.14461)
- Liu & Zhang (2018) — Surrogate residuals for ordinal regression
- Cheng, Wang & Zhang (2021) — Surrogate residuals for discrete choice models
- Greenwell et al. (2018) — the sure R package
- Barnard & Rubin (1999) — small-sample MI degrees of freedom
