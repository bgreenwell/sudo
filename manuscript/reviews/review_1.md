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

The principal unresolved issue is therefore learner-specific inference for a
cross-fitted machine-learning imputation model. Theorem M2 now gives
conditional convergence and variance consistency for the full-pipeline
bootstrap under a projected learner-stability condition. That condition is
not yet verified separately for GAM, MARS, or neural backfitting. The
fixed-imputation reference distribution and covariate-direction leakage
bound are now derived and numerically validated.

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

### 3. The black-box inference claim is now correctly delimited

The paper states that partially linear machine-learning imputation models
and a full-pipeline bootstrap attain nominal coverage in the designs studied.
The repository supports that statement for GAM, MARS, and neural
backfitting. Theorem M2 adds a learner-class result, but its projected
bootstrap-stability condition remains a learner-specific proof obligation,
so it does not support an unconditional general validity claim.

The abstract and contributions should use design-specific language, such as:

> In the black-box designs studied, a tuned partially linear imputation model
> paired with a full-pipeline bootstrap attained nominal coverage.

The discussion already contains much of the necessary qualification.

### 4. The machine-learning theory is high-level rather than learner-specific

Theorem M1 now treats the cross-fitted flexible imputation index
`v(D, X) = alpha D + f(X)` directly and retains the parametric formulas as a
special-case corollary. It isolates the first-order residual-weighted
projection of the fitted-index error and the fixed-imputation completion
noise. Theorem M2 supplies the corresponding bootstrap result under
conditional stability. The remaining gaps are learner-specific:

- prove the projected asymptotic-linearity condition for GAM, MARS, and
  neural backfitting;
- prove its conditional bootstrap analogue for those learners;
- verify the stated uniform remainder and conditional nuisance-rate
  conditions rather than assuming them at learner-class level;
- derive the correct fixed-imputation reference distribution.

Tang and Westling provide general bootstrap conditions for asymptotically
linear estimators with machine-learned nuisances
([arXiv:2404.03064](https://arxiv.org/abs/2404.03064)). Lin and Han establish
exchangeably weighted bootstrap validity for general DML estimators under the
conditions needed for DML itself
([arXiv:2604.17239](https://arxiv.org/abs/2604.17239)). Neither source, based
on its stated scope, discharges SUDO's projected learner condition or its
randomized generated-outcome continuity condition.

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

### 6. Fixed-imputation inference now has a reference law

The result that the scaled between-imputation variance retains a chi-square
fluctuation when the number of imputations is fixed is important. It implies
that increasing sample size alone does not make the reported variance
deterministic.

The joint limit now writes the estimator numerator as a normal variable
independent of Rubin's shifted-chi-square denominator. The exact reference
is therefore a normal-over-shifted-chi-square mixture, not generally a
Student distribution. A variance-calibrated Satterthwaite approximation
follows by matching the denominator's first two moments. Barnard-Rubin is
not exact for this law, but is close in the validated testbed.

At strong signal, predicted normal coverage is 0.888, 0.921, and 0.938 for
5, 10, and 25 imputations. Barnard-Rubin raises it to 0.942, 0.947, and
0.948; the calibrated Satterthwaite approximation stays within 0.006 of
0.95. Across both signals and all five imputation counts, predicted and
finite-sample coverage differ by at most 0.012. Barnard-Rubin with at least
25 imputations remains the practical default because the exact and
Satterthwaite references require additional component estimates.

Taking the number of imputations to infinity removes Monte Carlo uncertainty
in estimating the between-imputation component. It does not mean that every
variance contribution associated with imputation disappears.

### 7. The application caveat is repaired

The wine application clearly labels its causal interpretation as dependent
on a debatable graph and strong adjustment assumptions. The leave-one-out
analysis and two types of sensitivity calculations are useful.

The manuscript now reports the treatment-direction sensitivity multiplier
and conditions its interpretation explicitly on negligible
covariate-direction misspecification. Proposition A4 supplies the missing
exact covariate-leak identity and weighted norm bound, so readers can assess
a specified covariate perturbation separately. No general robustness claim
is made from the treatment-direction calculation alone.

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

7. **MARS support.** The cited continuous-treatment DML material did not
   validate the proposed `earth::linpreds` implementation, and inspection
   showed that `linpreds` alone still permits treatment interactions. The
   validated implementation instead builds bases from `X` only and estimates
   a mandatory linear treatment coefficient in a joint logistic fit.

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

Status: complete. `R/stage3s_pl_learners.R` compares GAM, MARS, and the tuned
PL-backfit on common data-generating seeds. MARS is tuned on theta-level
Monte Carlo bias and mechanically excludes treatment from its adaptive
basis. All three pass the bias and percentile-coverage criteria. At
$\theta_0=3$, their biases are 0.043, -0.010, and 0.007, respectively, and
their SD/mean-SE ratios are 0.960, 0.945, and 0.998. The comparison therefore
supports the partially-linear learner class rather than a unique backfitting
implementation.

Treat `v(D, X) = alpha D + f(X)` as the structural requirement, not
PL-backfit as the method. Compare:

- the existing logistic GAM with an unpenalized linear treatment term;
- MARS with treatment forced linear and basis selection restricted to `X`;
- the tuned PL-backfit already used by the full-pipeline bootstrap.

Use common samples, folds, seeds, and inference settings. Tune MARS on
theta-level Monte Carlo performance and promote it only if it passes the same
bias and coverage criteria as the existing default.

### Phase 3: Develop a learner-class asymptotic expansion

Status: the learner-class expansion is now stated as Theorem M1. It requires
an asymptotically linear residual-weighted projection of the fold-specific
imputation error and yields
`Var(e R + rho(W)) + sigma_u^2/B`, scaled by `J_theta^-2`.
`R/theory_ml_expansion.R` verifies the generated-outcome Taylor step for GAM,
MARS, and PL-backfit: the second-order ratio is 0.016 to 0.027 in every cell,
and the projected remainder is small relative to the first-order scale. The
remaining theoretical task is learner-specific verification of the projected
asymptotic-linearity condition, especially under bootstrap resampling.

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

Status: narrowed to a precise learner-class result. Theorem M2 establishes
conditional weak convergence around the bootstrap conditional mean and
bootstrap-variance consistency under projected learner stability,
conditional nuisance rates, integrated-map differentiability, randomized
completion continuity, and regular folds. The raw inverse-transform map need
not be differentiable for every fixed uniform. Fixed and redrawn folds are
both admissible under the stated condition, so redrawing is a propagation
choice rather than a necessity.

The fixed-imputation centering distinction is material. Fresh surrogate
randomness leaves an order-$n^{-1/2}$ difference between the realized
randomized estimate and the bootstrap conditional mean. The theorem
therefore supports the normal interval based on the full-pipeline bootstrap
SD, but not a conventional percentile or studentized interval without an
additional argument or increasing imputation count. Learner-specific
verification of projected bootstrap stability remains open.

The paired numerical check is complete for GAM, MARS, and neural backfitting
at both effect sizes. Fixed/redrawn mean-SE ratios range from 0.97 to 1.08,
with every paired bias difference inside its Monte Carlo tolerance. The
within-fold SE is 0.90 to 0.93 of the full-pipeline SE at moderate signal and
only 0.75 to 0.77 at strong signal. The improper SE is another 8% to 25%
smaller. The within-fold procedure therefore sits between improper and full
propagation rather than reproducing the improper variance exactly. Normal
coverage is 0.93 to 0.97; conventional studentization ranges from 0.80 to
0.93 and is not promoted.

Completed deliverables are the centered conditional law, bootstrap-variance
consistency, integrated-map argument, fixed/redrawn fold comparison, and
direct full/within/improper variance comparison. The remaining deliverable
is learner-specific verification of (MB1), with its uniform remainder and
conditional-rate requirements, for each implemented learner.

### Phase 5: Derive fixed-imputation inference

Status: complete. The fixed-$B$ joint law establishes independence between
the normal estimator numerator and shifted-chi-square denominator. The exact
mixture, calibrated Satterthwaite, Barnard-Rubin, and normal references are
compared at imputation counts 5, 10, 25, 50, and 100 under weak and strong
signal. `R/theory_fixed_b_inference.R` passes every theory-versus-finite
sample coverage check at $n=5000$ and 1,000 replications. The separate
full-pipeline comparison reports percentile and studentized intervals for
the machine-learning case and does not promote either over the normal
bootstrap-SD interval.

### Phase 6: Bound covariate-direction leakage

Status: complete. Proposition A4 expresses the exact finite-perturbation
leak through arm-specific path-averaged derivatives and gives both a global
weighted Cauchy-Schwarz bound and a sharper local second-order bound.
Overlap enters through `J_theta = E[m(X){1-m(X)}]`, with an explicit lower
bound under uniform overlap.

`R/theory_covariate_leakage.R` verifies both bounds across 12 interaction
cells using two million population draws. The local bound is 2.82 to 7.79
times the exact leak and the Taylor remainder scales as
`0.0027` to `0.0091` times the squared perturbation. The stage-7 interaction
arm is also decomposed into covariate leakage and induced
treatment-coefficient error, correcting the earlier claim that it was a
pure covariate-direction experiment.

### Phase 7: Complete secondary methodological work

After the inferential work:

- retune XGBoost using target-level Monte Carlo performance;
- measure the effective treatment-direction error directly across learners;
- leave multinomial outcomes as a separate project.

## Recommended execution order

Items 1 through 6 in the original sequence are complete. The remaining
order is:

1. verify the projected asymptotic-linearity and bootstrap-stability
   conditions separately for GAM, MARS, and neural backfitting;
2. retune or directly diagnose secondary learner classes only after that
   proof attempt identifies what stability must be measured;
3. treat multinomial outcomes as a separate project.
