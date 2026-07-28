# AGENTS.md

Context for agents and human contributors working in this repository. Read
this before making changes.

## Project

**SUDO** (Surrogate-Assisted Double Machine Learning) is causal inference for
binary and ordinal outcomes in the DML partially linear model. A discrete
outcome `Y` is completed to a continuous latent-utility *surrogate* (a
truncated inverse-transform draw from the assumed link law); FWL
partialling-out runs on each completed dataset; and `B` proper
multiple-imputation draws are pooled with Rubin's rules. The theory notes are
in the paper's asymptotic-theory appendix; the paper is in
`manuscript/paper/`.

## Workflow: R first, then Python

Every methodological change is prototyped and validated in an `R/stageN_*.R`
script with explicit acceptance criteria (bias against Monte-Carlo standard
error, coverage bands) before it is ported to the Python package. Do not add
method code to Python that has no validated R stage behind it. Each stage
script is standalone, is run from the repository root, and asserts its own
pass conditions.

## Setup and commands

```bash
# R validation ladder (run from repo root; each asserts its own pass)
Rscript R/stage0_fwl.R                 # FWL exactness, cross-fit PLR sanity, DoubleML check
Rscript R/stage1_binary_surrogate.R    # single-draw binary surrogate FWL
Rscript R/stage2_binary_rubin.R        # Rubin pooling: naive vs improper vs proper MI
Rscript R/stage3_binary_dml.R          # full binary SUDO, ML nuisances (parallel, ~10 min)
Rscript R/stage3m_pl_backfit_tuning.R  # tuned partially-linear backfit (bias fix)
Rscript R/stage3p_pipeline_bootstrap.R # full-pipeline bootstrap (SE fix, black-box model)
Rscript R/stage4_ordinal_simple.R      # ordinal J=3, clm full model
Rscript R/stage5_ordinal_dml.R         # ordinal, spline clm, gam nuisances (parallel)
Rscript R/stage5r_ordinal_refit.R      # corrected ordinal per-draw outcome nuisance
Rscript R/stage6_link_misspec.R        # cloglog truth vs logit analyst, plus diagnostics
Rscript R/wine_application.R           # application: volatile acidity on ordinal wine quality
Rscript R/wine_sensitivity.R           # adjustment-set robustness and omitted-variable bound

# Theory checks behind the paper's asymptotic-theory appendix
Rscript R/theory_ordinal_passthrough.R # pass-through derivatives, J=2,3,4, three links
Rscript R/theory_variance_terms.R      # fixed-B variance and the Rubin exactness gap
Rscript R/theory_ordinal_variance_terms.R # ordinal threshold and Rubin terms
Rscript R/theory_ordinal_nuisance_ladder.R # isolate ordinal variance inflation

# Python package (always use uv from python/; no conda or system Python)
cd python && uv sync && uv run pytest

# Paper
cd manuscript/paper && quarto render sudo_paper.qmd --to arxiv-pdf
```

The `R/` directory holds the full ladder; the commands above are the load-
bearing stages. Additional `stage3*` scripts explore the black-box
imputation-model program and are documented inline.

## Repository layout

```
sudo/
├── R/
│   ├── sudo/        # sourced helpers: surrogate.R, rubin.R, fwl.R, mc.R, estimator.R
│   ├── stage*.R     # the validation ladder, simple to complex
│   └── results/     # committed stage summary CSVs
├── python/          # package mirroring the validated R design
│   ├── src/sudo/
│   └── tests/
├── manuscript/
│   ├── paper/       # Quarto arxiv draft and references.bib
│   └── data/wine/   # UCI wine-quality data for the application
├── AGENTS.md  CLAUDE.md  README.md  LICENSE
```

R scripts source helpers with `source("R/sudo/...")` and must be run from the
repository root.

## Conventions

- **Commits.** Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`,
  `test:`, `chore:`, `ci:`), optionally scoped (`fix(ansi): ...`). Imperative
  mood, first line under 72 characters, issue/PR references in parentheses at
  the end. Do not squash-merge PRs.
- **Prose.** No em dashes, no emoji, no filler. Describe the user-visible
  effect, not the implementation. This applies to code comments, commit
  messages, and docs.
- **CHANGELOG / NEWS.** Keep a Changelog format, one line per entry, standard
  sections only (Added / Changed / Deprecated / Removed / Fixed / Security).
- **Python.** Run everything through `uv` from `python/`.
- **R.** Scripts are run from the repository root and assert their own
  acceptance criteria.

## Key design decisions

- **Proper MI draws.** Each surrogate draw first redraws the full-model
  parameters from `N(beta_hat, vcov)` (thresholds included for ordinal: use
  `ordinal::clm`, whose vcov covers them; `mgcv::ocat` does not). Improper
  draws under-cover (about 0.93); stage 2 quantifies this.
- **Full model includes D.** It estimates `P(Y | D, X)`; the outcome nuisance
  `E[S | X]` strips `D` back out; cross-fitting prevents leakage.
- **Nuisances fit once.** Only `S` and the parameter draw are redrawn across
  the `B` iterations (stage 3b shows refitting changes nothing).
- **Link conventions.** Binary glm cloglog is a Gumbel-max latent error;
  ordinal clm cloglog is Gumbel-min. `complete_surrogate` handles both.
- **Diagnostics.** Surrogate-residual variance drift in `X` (via `sure` plus
  mgcv) detects link misspecification; mean-structure tests are blind to it.
  Diagnostic draws use an RNG stream independent of the estimation draws.
- **Estimand.** `theta` is on the latent-utility scale, not the
  probability-scale ATE.
- **Full-model quality matters more than nuisance quality.** Orthogonality
  protects `theta` from error in the partialling nuisances, not from error in
  the full imputation model, whose error is baked into `S` itself. Full
  theta-level Monte-Carlo validation is therefore not optional for a
  black-box imputation model; no cheap diagnostic reliably predicts its bias.
- **Partially-linear (PL) full models beat every diagnostic-driven fix**
  (stages 3j to 3l): forcing `V = beta*D + f(X)` so `D` cannot interact with
  `X`. Backfitting (IRLS, `f` via `nnet`) is the strongest result of the
  black-box program. A "principled" cross-validation tuning procedure can be
  actively harmful if it optimizes the wrong objective (mboost's `cvrisk`
  minimizes deviance, not D-coefficient bias, and is the worst arm): any
  tuning search must select on theta-level MC performance directly.
- **Truncation pass-through** (paper appendix, Prop A1). Index error
  reaches `theta` with factor `c(e) = 1 - dlogis(-e)*(mu_+(e) - mu_-(e))`,
  rising from about 0.31 to 1 as the signal grows, so
  `bias ~= cbar_1*delta_D + X-leak` with `cbar_1 = 0.669` (theta = 1.5) and
  `0.861` (theta = 3) under the stage-3 DGP. The truncation self-corrects
  because `Y` comes from the truth, but that help vanishes once `Y` is nearly
  deterministic; this, not quasi-separation, is why every learner degrades at
  theta = 3. The X-direction is not fully protected either (`S` is nonlinear
  in `V_hat`), only small here.
- **A black-box imputation model can give valid inference.** Two pieces: the
  tuned PL-backfit (stage 3m, `size=4, decay=0.3, n_iter=5`, selected on MC
  theta bias) fixes the bias; the full-pipeline bootstrap
  (`sudo_pipeline_boot`, stage 3p) fixes the SE (theta = 3 coverage 0.860 to
  0.95). A recentered within-fold bootstrap under-propagates full-model
  variance; resampling the whole dataset and rerunning the pipeline captures
  it.
- **Watch the calibration-slope direction.** Stages 3g and 3j to 3l regress
  `eta ~ V_hat`, so slope above 1 means `V_hat` is attenuated, below 1 means
  inflated.
- **Never summarize a signed diagnostic with `mean(abs(.))`.** Stage 3h's
  early "negative" verdict was purely that artifact (see
  `stage3h_reanalysis.R`).

## Gotchas

- **Parallel MC.** mgcv is not fork-safe on macOS. Use `mc_cluster()` /
  `run_mc_par()` (PSOCK) from `R/sudo/mc.R`, and build dgp/estimator closures
  with `force()`-ed factory functions, never bare top-level closures.
