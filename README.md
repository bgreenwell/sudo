# SUDO: Surrogate-Assisted Double Machine Learning

A uniform completion-and-projection interface for discrete outcomes in the
double machine learning (DML) partially linear model.

## The problem

Suppose you have run a study with a discrete outcome, a yes/no event, a
rated response on a 1 to 5 scale, or a count, and you want the effect of a
treatment while adjusting flexibly for many covariates.

For a continuous outcome, DML is the standard tool: predict the outcome from
the covariates, predict the treatment from the covariates, subtract off both
predictions, and read the treatment effect from what is left. Subtracting
first is what makes it safe to use flexible machine-learning models for the
two predictions without biasing the effect.

That machinery assumes the observed outcome is on the additive target scale.
Running it directly on a discrete label generally targets a different
quantity. Binary and ordinal cumulative-link models are additive on a latent
utility scale; Poisson and negative-binomial models are additive on the log
conditional-mean scale. Tailored orthogonal scores can be derived for a
chosen law. SUDO supplies one common interface across laws.

## The idea

For a fitted discrete CDF $G$, SUDO draws uniformly between
$G(Y-1\mid D,X)$ and $G(Y\mid D,X)$, maps the draw through a mean-zero
reference quantile, and adds the fitted scalar index. Ordinary
Frisch-Waugh-Lovell partialling-out then applies to the continuous
completion. Cumulative links use their latent error law; counts use a
standard-normal randomized-quantile reference.

For parametric binary and ordinal models, the randomized completions are
multiple imputations, so `B` of them are pooled with Rubin's rules. These
draws have to be *proper*: the imputation model's own parameters are redrawn
before each completion, thresholds included. Flexible fitted learners instead
use a full-pipeline bootstrap to propagate model uncertainty. Counts also
admit an analytic Rao-Blackwell completion that integrates out the randomized
reference draw.

For cumulative-link models, this randomized completion is the latent
surrogate response introduced for model diagnostics by
[Liu and Zhang](https://doi.org/10.1080/01621459.2017.1292915) and implemented
in [`sure`](https://github.com/bgreenwell/sure). SUDO does not introduce that
response. It uses the response as an outcome for treatment-effect estimation,
removes the part explained by `X` while retaining the treatment component,
and adds cross-fitting and uncertainty propagation. Stage 20 verifies exact
seeded agreement for binary logit/probit and ordinal logit/probit/cloglog,
after accounting for `sure`'s ordinal threshold-origin normalization.

The current local `sure` development version is an exception for binary
cloglog: it selects Gumbel-min, while a binomial cloglog GLM requires
Gumbel-max. SUDO uses the required binary convention. The discrepancy is
documented in [`sure` issue 45](https://github.com/bgreenwell/sure/issues/45).

Two caveats are structural and worth stating up front. The true index must
have the form $\eta_0(D,X)=\theta_0 D+g_0(X)$. A fitted imputation learner
need only supply a pointwise scalar index and usable conditional CDF; it need
not expose a treatment coefficient. The two adjustment
models are protected by orthogonality, but the imputation model is not: its
error is baked into the completed data rather than differenced away, so it
reaches the estimate at first order. The estimator is DML-assisted rather
than fully orthogonal, and the imputation model has to be right.

- Binary, ordinal, and count laws behind one randomized-CDF interface.
- Proper multiple-imputation draws for valid coverage, with a built-in link
  diagnostic from the surrogate residuals.
- A closed-form account of how imputation-model error reaches the estimate,
  and a full-pipeline bootstrap for black-box imputation models.

SUDO does not claim an efficiency or estimation advantage over a correct
tailored score or direct structured-model coefficient. It equals an
index-only projection plus an outcome-informed correction that can help,
hurt, or vanish. A structured coefficient is a special case. Binary
experiments validate the interface. Ordinal is the principal established
application. Count and flexible-ordinal support remain R-only until their
fitted-learner confirmatory stages pass. A six-cell oracle count bridge has
passed its bias, coverage, SD-to-SE, and analytic-versus-randomized
completion gates; this validates the count construction, not a deployable
fitted count learner.

## Installation

Python package (from `python/`, using [uv](https://docs.astral.sh/uv/)):

```bash
cd python && uv sync
```

R scripts need `ordinal`, `mgcv`, `mboost`, `splines`, `nnet`, `MASS`,
`earth`, `statmod`, `xgboost`, and `ranger`. Stage 20 also needs `pkgload` and
a local checkout of `sure`; set `SURE_SOURCE` if it is not at `../../r/sure`.

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
Rscript R/theory_pl_series.R           # series expansion and bootstrap theorem checks
Rscript R/theory_ml_expansion.R        # learner-class expansion checks
Rscript R/theory_ml_bootstrap.R        # full-pipeline bootstrap comparison
Rscript R/theory_fixed_b_inference.R   # fixed-B reference-law comparison
Rscript R/theory_covariate_leakage.R   # covariate leakage bound
Rscript R/stage4_ordinal_simple.R      # ordinal J=3, clm full model
Rscript R/stage5_ordinal_dml.R         # ordinal with flexible nuisances
Rscript R/stage5r_ordinal_refit.R      # corrected ordinal per-draw outcome nuisance
Rscript R/stage6_link_misspec.R        # wrong-link bias and variance-drift diagnostic
Rscript R/stage8_comparators.R         # direct ordinal coefficients alongside SUDO
Rscript R/stage9_penalty_path.R        # coefficient-plus-correction identity
Rscript R/stage11_debiased_comparison.R # tailored orthogonal-score comparison
Rscript R/stage12_link_flexibility.R   # correct cloglog score and link sensitivity
Rscript R/theory_discrete_completion.R # randomized-PIT centering across laws
Rscript R/stage13_binary_cloglog.R     # flexible non-logit binary confirmation
Rscript R/stage14t_ordinal_tuning.R    # target-level tuning on dedicated seeds
Rscript R/stage14_ordinal_flexible.R   # flexible ordinal confirmation
Rscript R/stage20_sure_equivalence.R   # latent-surrogate parity with local sure
Rscript R/theory_count_completion.R    # analytic count completion and edge checks
Rscript R/stage15o_count_oracle.R      # passed six-cell count oracle bridge
Rscript R/stage15t_count_tuning.R      # dedicated-seed count learner tuning
Rscript R/stage15_count.R              # coefficientless count confirmation and ablations
Rscript R/wine_application.R           # application: volatile acidity on wine quality
Rscript R/bikeshare_application.R      # count application with block bootstrap
```

Each script prints its Monte-Carlo table, asserts its pass criteria, and
writes a summary CSV to `R/results/`.

## Documentation

- `manuscript/paper/` holds the Quarto manuscript (`sudo_paper.qmd`), which
  renders to an arXiv PDF and generates its result tables from `R/results/`.
  Its asymptotic-theory appendix carries the pass-through map, the moment
  properties, the fixed-`B` variance, the pooled-variance analysis, and
  expansion and bootstrap results for a deterministic-series reference
  learner, proved under explicit series conditions. The adaptive-learner
  counterparts hold under high-level conditions that have not been verified
  for the learners actually implemented.
- `AGENTS.md` documents the workflow, layout, and key design decisions.
- `TODO.md` records the current validation and paper-development roadmap.

## Development

- R-first: do not add method code to Python without a validated R stage
  behind it. See `AGENTS.md` and `CONTRIBUTING.md`.
- Commits follow Conventional Commits; PRs are merged, not squashed.

## License

MIT. See `LICENSE`.
