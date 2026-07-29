# SUDO: Surrogate-Assisted Double Machine Learning

Causal inference for binary and ordinal outcomes in the double machine
learning (DML) partially linear model.

## The problem

Suppose you have run a study with a categorical outcome, a yes/no event or a
rated response on a 1 to 5 scale, and you want the causal effect of a
treatment while adjusting flexibly for many covariates.

For a continuous outcome, DML is the standard tool: predict the outcome from
the covariates, predict the treatment from the covariates, subtract off both
predictions, and read the treatment effect from what is left. Subtracting
first is what makes it safe to use flexible machine-learning models for the
two predictions without biasing the effect.

That machinery assumes a continuous outcome. A categorical outcome obeys the
additive structure only on a *latent* scale: there is an unobserved
continuous utility, and you observe which interval it fell into. Existing
tooling covers the binary-logistic case and nothing else.

## The idea

SUDO fills the latent outcome in. Each observation is completed to a
continuous *surrogate*, drawn from the assumed error distribution truncated
to the interval the observed category implies. Ordinary
Frisch-Waugh-Lovell partialling-out then applies to the completed data.

The completions are multiple imputations, so `B` of them are pooled with
Rubin's rules. The draws have to be *proper*: the imputation model's own
parameters are redrawn before each completion, thresholds included. Reusing
one fitted model across every draw hides its sampling error, and the
intervals come up short.

One caveat is structural and worth stating up front. The two adjustment
models are protected by orthogonality, but the imputation model is not: its
error is baked into the completed data rather than differenced away, so it
reaches the estimate at first order. The estimator is DML-assisted rather
than fully orthogonal, and the imputation model has to be right.

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
Rscript R/stage3s_pl_learners.R        # partially-linear learner comparison
Rscript R/theory_ml_expansion.R        # learner-class expansion checks
Rscript R/theory_ml_bootstrap.R        # full-pipeline bootstrap comparison
Rscript R/theory_fixed_b_inference.R   # fixed-B reference-law comparison
Rscript R/theory_covariate_leakage.R   # covariate leakage bound
Rscript R/stage4_ordinal_simple.R      # ordinal J=3, clm full model
Rscript R/stage5_ordinal_dml.R         # ordinal with flexible nuisances
Rscript R/stage5r_ordinal_refit.R      # corrected ordinal per-draw outcome nuisance
Rscript R/stage6_link_misspec.R        # wrong-link bias and variance-drift diagnostic
Rscript R/wine_application.R           # application: volatile acidity on wine quality
```

Each script prints its Monte-Carlo table, asserts its pass criteria, and
writes a summary CSV to `R/results/`.

## Documentation

- `manuscript/paper/` holds the Quarto manuscript (`sudo_paper.qmd`), which
  renders to an arXiv PDF and generates its result tables from `R/results/`.
  Its asymptotic-theory appendix carries the pass-through map, the moment
  properties, the fixed-`B` variance, and the pooled-variance analysis.
- `AGENTS.md` documents the workflow, layout, and key design decisions.

## Development

- R-first: do not add method code to Python without a validated R stage
  behind it. See `AGENTS.md` and `CONTRIBUTING.md`.
- Commits follow Conventional Commits; PRs are merged, not squashed.

## License

MIT. See `LICENSE`.
