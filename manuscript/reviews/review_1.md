# Review of SUDO manuscript, open problems, and external assessment

## Scope

This review covers the full paper and appendices in
`manuscript/paper/sudo_paper.qmd`, the research agenda in
`manuscript/open_problems.md`, the committed simulation summaries, and the
first external assessment previously stored in this file. It checks the
external assessment against the manuscript, repository evidence, and primary
or authoritative sources.

## Overall assessment

SUDO has a credible methodological core, an original use of surrogate
residuals as latent-outcome completions, and an unusually transparent account
of its limitations. The pass-through analysis is insightful, the numerical
verification is stronger than is typical for a paper at this stage, and the
R-first validation ladder makes the empirical claims auditable.

The ordinal machine-learning variance anomaly identified in the first review
has now been isolated and corrected: the outcome nuisance must be refit for
each ordinal completion. The corrected estimator passes its Monte
Carlo-aware bias, coverage, and variance-calibration criteria and is now the
R and Python default. The paper also states the conditional latent-error law
explicitly, distinguishes SUDO from existing logistic DML, and qualifies the
black-box evidence by design.

The principal unresolved issue is therefore inference for a cross-fitted
machine-learning imputation model. The full-pipeline bootstrap remains an
empirical result rather than a proved procedure, and fixed-imputation
reference distributions and covariate-direction leakage still need formal
treatment.

The previous external assessment was useful mainly as an agenda check. It
correctly identified several open areas already recorded in
`manuscript/open_problems.md`, but it was not reliable enough to serve as a
technical review. It used broken mathematical image references, relied on
weak or unrelated sources, and often converted conjectures into conclusions.
Its useful suggestions are incorporated below only where they survive
independent review.

## Major findings

### 1. Conditional-law identification assumptions are repaired

The surrogate identity requires the latent-error distribution conditional on
the observed design to agree with the assumed link law. The displayed model
previously said

```text
epsilon ~ F
```

while selection on observables is expressed through conditional mean
independence. That combination does not by itself state the full condition
needed for the completion argument. The required assumption is closer to

```text
epsilon | D, X ~ F
```

with the appropriate location, scale, and threshold normalization.

The formal setup and Assumption A1 now state the conditional law and
separate:

1. latent ignorability or causal mean independence;
2. the conditional location-family model for the latent error;
3. overlap;
4. a constant latent-scale treatment effect;
5. threshold and scale normalization.

This completed clarification is foundational. The identity
`E[S | D, X] = E[U | D, X]`, the link diagnostic, and the truncation
derivatives all depend on it.

### 2. The logistic-DML comparison is repaired

The manuscript no longer says that SUDO subsumes the logistic partially
linear estimator of Liu, Zhang, and Zhou. It now says that the methods target
the same latent-logit coefficient under the logit specification and presents
SUDO as a link-flexible alternative. No estimator equivalence or common
efficiency claim is made.

### 3. The black-box inference claim exceeds the evidence

The paper states that a partially linear machine-learning imputation model
and a full-pipeline bootstrap attain nominal coverage. The repository
supports that statement in the black-box designs studied, with 100 outer
replications. It does not yet support a general validity claim.

The abstract and contributions should use design-specific language, such as:

> In the black-box designs studied, a tuned partially linear imputation model
> paired with a full-pipeline bootstrap attained nominal coverage.

The discussion already contains much of the necessary qualification.

### 4. The machine-learning imputation theory is not theorem-ready

The appendix appropriately describes its parametric results as derivations
with proof sketches and numerical verification. Those formulas are useful as
a special case and benchmark, but the methodological center of the paper is
a cross-fitted flexible imputation index of the form
`v(D, X) = alpha D + f(X)`. The missing result is a learner-class expansion
for that estimator. The remaining gaps include:

- an expansion of the residual-weighted, fold-specific imputation error;
- uniform control of stochastic remainders;
- conditional nuisance-rate requirements under bootstrap resampling;
- differentiability of the generated-outcome map;
- the correct fixed-imputation reference distribution;
- learner-specific stability conditions for the full-pipeline bootstrap.

Tang and Westling provide general bootstrap conditions for asymptotically
linear estimators with machine-learned nuisances
([arXiv:2404.03064](https://arxiv.org/abs/2404.03064)). Lin and Han establish
exchangeably weighted bootstrap validity for general DML estimators under the
conditions needed for DML itself
([arXiv:2604.17239](https://arxiv.org/abs/2604.17239)). Neither source, based
on its stated scope, directly proves validity for SUDO's randomized generated
outcome.

Hahn and Ridder study the influence of first-step generated regressors in
three-step semiparametric estimators
([Econometrica 81, 315-340](https://doi.org/10.3982/ECTA9609)). Their
path-derivative approach may help construct SUDO's influence-function
expansion, but it is not itself a bootstrap theorem for SUDO.

### 5. Ordinal over-coverage is resolved

The original implementation confirmed substantial conservatism:

- stage 4 reports coverage of 0.976;
- stage 5 at `n = 2000` reports SD 0.110, mean SE 0.149, and coverage 0.985;
- the internal Rubin checks report variance ratios of roughly 1.50 to 1.88.

The ordinal term-level decomposition then ruled out threshold estimation and
pooling alone. The paired nuisance ladder isolated reuse of one plug-in
estimate of `E[S | X]` across completions: per-draw refitting left empirical
variance nearly unchanged while reducing `T/V` from 1.505 to 0.942. The full
corrected stage at `n = 2000`, `B = 25`, and 200 replications reports bias
0.016, coverage 0.960, and `T/V = 1.214` with Monte Carlo SE 0.122. It passes
all prespecified checks.

The remaining sample-size and imputation-count grid belongs with the
fixed-imputation investigation. It does not block the corrected default.

### 6. Fixed-imputation inference is both a contribution and an open issue

The result that the scaled between-imputation variance retains a chi-square
fluctuation when the number of imputations is fixed is important. It implies
that increasing sample size alone does not make the reported variance
deterministic.

The paper correctly stops short of proving that the Barnard-Rubin degrees of
freedom give the right reference distribution for SUDO. The original
Barnard-Rubin paper derives an adjustment for finite complete-data degrees of
freedom in a Bayesian multiple-imputation framework
([Biometrika 86, 948-955](https://doi.org/10.1093/biomet/86.4.948)).
The stronger claim in the previous assessment that its derivation
specifically assumes independence of SUDO's within- and between-imputation
components was not verified from that source and should not be repeated
without a precise derivation.

Taking the number of imputations to infinity removes Monte Carlo uncertainty
in estimating the between-imputation component. It does not mean that every
variance contribution associated with imputation disappears.

### 7. The application is appropriately cautious, with one overstatement

The wine application clearly labels its causal interpretation as dependent
on a debatable graph and strong adjustment assumptions. The leave-one-out
analysis and two types of sensitivity calculations are useful.

The conclusion that the imputation model would have to be wrong by an amount
that is "not credible" is stronger than the analysis supports. That
calculation isolates treatment-direction error, while the paper demonstrates
that covariate-direction leakage can also be material. The paper should
report the multiplier and either let readers assess plausibility or condition
the conclusion explicitly on negligible covariate-direction misspecification.

## Assessment of the previous external review

### Points worth retaining

The previous review correctly emphasized:

- full-pipeline bootstrap validity as an open theorem;
- the generated-outcome step as distinct from ordinary DML;
- an ordinal term-level variance decomposition;
- threshold contributions to the derivative;
- uncertainty about fixed-imputation reference distributions;
- the absence of theory for cross-fitted flexible imputation models;
- MARS as a possible partially linear learner.

Most of these points were already present in `manuscript/open_problems.md`,
but they remain valid priorities.

### Points to reject or restate

1. **Six bootstrap conditions as a theorem.** The listed conditions are a
   plausible checklist, not established sufficient conditions for SUDO.

2. **Mandatory redrawing of folds.** Redrawing folds within each bootstrap
   resample may be a sound implementation choice. Its necessity has not been
   proved. Fixed and redrawn folds should be compared directly.

3. **Threshold derivatives as the ordinal solution.** The implementation
   already draws thresholds jointly, and Proposition A1 already includes the
   threshold derivative block. The open question is whether the resulting
   terms explain the variance gap.

4. **Mandatory Firth or ridge estimation.** Near-separation deserves a
   regularity analysis, but the review did not establish that bias reduction
   or ridge must always be used. Bias-reduced cumulative-link estimation is
   relevant background
   ([Kosmidis, arXiv:1204.0105](https://arxiv.org/abs/1204.0105)), not proof
   of necessity for SUDO.

5. **Between-imputation variance vanishes as the number of imputations
   grows.** What vanishes is Monte Carlo error from estimating the component,
   not necessarily the component itself.

6. **XGBoost fails because trees are nonsmooth.** This mechanism was asserted
   without evidence. The repository's tuning-quality hypothesis is more
   defensible and directly testable.

7. **MARS support.** The cited continuous-treatment DML material does not
   validate the proposed `earth::linpreds` implementation. MARS should be
   treated as an experimental backlog item.

8. **Source quality.** ResearchGate pages, Medium posts, software
   documentation, unrelated applications, and a CRAN task view are not
   adequate support for the review's theoretical claims. The broken
   `![][imageN]` expressions also made its mathematics unauditable.

## Prioritized plan

### Phase 1: Reconcile the completed ordinal correction

Status: the first isolation step is complete. The new
`R/theory_ordinal_variance_terms.R` stage includes the threshold block and
passes its Monte Carlo acceptance checks. It predicts `T/V = 1.028` at
moderate signal and `0.980` at strong signal, ruling out ordinal threshold
estimation and Rubin pooling alone as explanations for the stage-5 ratio of
1.50 to 1.88.

The paired nuisance ladder is also complete at 100 replications and `B = 10`.
It identifies reuse of a fixed plug-in outcome nuisance as the mechanism:
per-draw refitting changes empirical variance from 0.01380 to 0.01375 but
reduces mean Rubin variance from 0.02077 to 0.01295. The corresponding `T/V`
falls from 1.505 to 0.942.

The full corrected stage is now complete at `n = 2000`, `B = 25`, and 200
replications. It reports bias 0.016, coverage 0.960, and `T/V = 1.214` with
Monte Carlo SE 0.122. It passes its bias, coverage, and variance-calibration
criteria, so per-draw outcome-nuisance refitting is now the validated ordinal
default.

Remaining work is to remove stale descriptions of the anomaly and to fold the
additional sample-size and imputation-count grid into fixed-imputation work.

### Phase 2: Compare structurally partially linear learners

Treat `v(D, X) = alpha D + f(X)` as the structural requirement, not
PL-backfit as the method. Compare:

- the existing logistic GAM with an unpenalized linear treatment term;
- MARS with treatment forced linear and basis selection restricted to `X`;
- the tuned PL-backfit already used by the full-pipeline bootstrap.

Use common samples, folds, seeds, and inference settings. Tune MARS on
theta-level Monte Carlo performance and promote it only if it passes the same
bias and coverage criteria as the existing default.

### Phase 3: Develop a learner-class asymptotic expansion

Define the imputation estimator by fold-specific functions
`v_hat[-k](D, X) = alpha_hat[-k] D + f_hat[-k](X)`, without requiring a
finite-dimensional parameterization.

Required deliverables:

- a precise estimator definition including auxiliary randomness;
- an influence-function expansion for fixed imputation count;
- a projected expansion for
  `E[R {psi(v_hat) - psi(v0)}]`;
- a lemma controlling its interaction with cross-fitted adjustment
  nuisances;
- uniform remainder conditions;
- a clear distinction between conditional-on-imputation and unconditional
  laws;
- the existing parametric formula as a special-case corollary.

Acceptance criterion: a complete proof that does not assume the desired
expansion or bootstrap consistency as a hypothesis.

### Phase 4: Prove or narrow full-pipeline bootstrap validity

Map the ordinary asymptotically linear and DML components to the conditions
in Tang-Westling and Lin-Han. Prove the SUDO-specific generated-outcome
result separately.

Questions to settle:

- whether differentiability is needed for the randomized quantile map, its
  conditional expectation, or only the integrated score;
- whether common random numbers give a useful smooth representation away
  from probability boundaries;
- whether redrawn folds are necessary or only sufficient;
- what the bootstrap target is when the imputation count is fixed;
- which projected-error stability conditions hold for GAM, MARS, and
  PL-backfit.

Acceptance criteria:

- conditional weak convergence to the same fixed-imputation law;
- a numerical comparison of fixed and redrawn folds;
- a direct test of the claim that the within-fold bootstrap reproduces the
  improper-draw variance deficit.

### Phase 5: Derive fixed-imputation inference

Characterize the joint limiting law of the pooled estimator, mean within
variance, and between-imputation variance. Compare:

- Barnard-Rubin intervals;
- a Satterthwaite approximation based on the derived covariance;
- normal inference with an increasing imputation count;
- percentile and studentized full-pipeline bootstrap intervals.

Acceptance criterion: theory-predicted and empirical coverage across
imputation counts 5, 10, 25, 50, and 100 under weak and strong signal.

### Phase 6: Bound covariate-direction leakage

Derive a weighted norm bound involving overlap,
`c(eta_1) - c(eta_0)`, and the covariate-direction index error.

Acceptance criterion: a bound that is computable or conservatively
estimable and has the correct order in the interaction-misspecification
experiment.

### Phase 7: Complete secondary methodological work

After the inferential work:

- retune XGBoost using target-level Monte Carlo performance;
- measure the effective treatment-direction error directly across learners;
- leave multinomial outcomes as a separate project.

## Recommended execution order

The immediate sequence should be:

1. reconcile the completed ordinal correction and remaining paper claims;
2. compare GAM, MARS, and PL-backfit under a common design;
3. develop the learner-class asymptotic expansion;
4. prove or narrow the full-pipeline bootstrap result;
5. derive the fixed-imputation reference law;
6. bound covariate-direction leakage;
7. leave other learners and extensions as secondary work.

This order gives the fastest opportunity either to validate the current
inferential account or to identify precisely where the estimator, variance,
or manuscript claims must change.
