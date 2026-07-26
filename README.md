# SUDO: Surrogate-Assisted Double Machine Learning

Causal inference for binary and ordinal outcomes in the double machine
learning (DML) partially linear model.

Binary and ordinal outcomes have no home in the DML partially linear model:
the additive structure lives on a latent scale, and existing tooling covers
only the binary-logistic case. SUDO completes the discrete outcome to a
continuous latent-utility surrogate (a truncated inverse-transform draw from
the assumed link law), runs standard Frisch-Waugh-Lovell partialling-out on
each completed dataset, and pools B proper multiple-imputation draws with
Rubin's rules.

- Binary and ordinal outcomes in one framework, not tied to the logit link.
- Proper multiple-imputation draws for valid coverage, with a built-in link
  diagnostic from the surrogate residuals.
- A closed-form account of how imputation-model error reaches the estimate,
  and a full-pipeline bootstrap for black-box imputation models.

## Installation

Python package (from `python/`, using [uv](https://docs.astral.sh/uv/)):

```bash
cd python && uv sync
```

R scripts need `ordinal`, `mgcv`, `splines`, `nnet`, and `MASS`.

## Quick start

```python
from sudo import SudoDML, generate_binary

X, D, y, _ = generate_binary(2000, theta=1.5, random_seed=1)
model = SudoDML(B=25, random_state=1).fit(X, D, y)
print(model.theta_, model.se_, model.ci_)
```

## Validation ladder

Development is R-first: each methodological step is a standalone script with
explicit acceptance criteria, validated before being ported to the Python
package. Run from the repository root.

```bash
Rscript R/stage0_fwl.R                 # FWL exactness; cross-fit PLR sanity
Rscript R/stage1_binary_surrogate.R    # binary surrogate FWL, single draw
Rscript R/stage2_binary_rubin.R        # Rubin pooling: naive vs improper vs proper MI
Rscript R/stage3_binary_dml.R          # full binary SUDO with ML nuisances
Rscript R/stage4_ordinal_simple.R      # ordinal J=3, clm full model
Rscript R/stage5_ordinal_dml.R         # ordinal with flexible nuisances
Rscript R/stage6_link_misspec.R        # wrong-link bias and variance-drift diagnostic
Rscript R/wine_application.R           # application: volatile acidity on wine quality
```

Each script prints its Monte-Carlo table, asserts its pass criteria, and
writes a summary CSV to `R/results/`.

## Documentation

- `manuscript/paper/` holds the Quarto manuscript (`sudo_paper.qmd`), which
  renders to an arXiv PDF and generates its result tables from `R/results/`.
- `manuscript/sudo.md` holds the theory notes, ladder-ordered.
- `AGENTS.md` documents the workflow, layout, and key design decisions.

## Development

- R-first: do not add method code to Python without a validated R stage
  behind it. See `AGENTS.md` and `CONTRIBUTING.md`.
- Commits follow Conventional Commits; PRs are merged, not squashed.

## License

MIT. See `LICENSE`.
