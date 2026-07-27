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

The paper is not yet ready to support its strongest inferential and novelty
claims. The most visible unresolved issue is the substantial over-coverage in
the ordinal machine-learning design. The full-pipeline bootstrap also remains
an empirical result rather than a proved procedure. In addition, the formal
identification assumptions should state the required conditional latent-error
law more precisely, and several claims about existing logistic DML and
black-box validity should be narrowed.

The previous external assessment was useful mainly as an agenda check. It
correctly identified several open areas already recorded in
`manuscript/open_problems.md`, but it was not reliable enough to serve as a
technical review. It used broken mathematical image references, relied on
weak or unrelated sources, and often converted conjectures into conclusions.
Its useful suggestions are incorporated below only where they survive
independent review.

## Major findings

### 1. State the identification assumptions as conditional laws

The surrogate identity requires the latent-error distribution conditional on
the observed design to agree with the assumed link law. The displayed model
currently says

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

The prose recognizes that the conditional law, not merely its mean, must be
correct. The formal setup and Assumption A1 should say the same thing. The
following components should be separated:

1. latent ignorability or causal mean independence;
2. the conditional location-family model for the latent error;
3. overlap;
4. a constant latent-scale treatment effect;
5. threshold and scale normalization.

This clarification is foundational. The identity
`E[S | D, X] = E[U | D, X]`, the link diagnostic, and the truncation
derivatives all depend on it.

### 2. The claim that SUDO subsumes logistic DML is too strong

The manuscript describes the logistic partially linear estimator of Liu,
Zhang, and Zhou as a special case subsumed by SUDO. Both methods may target
the same latent-logit coefficient, but the paper does not establish that the
surrogate-completion estimator and the logistic orthogonal-score estimator
are identical, asymptotically equivalent, or members of the same efficiency
class.

Unless that equivalence is proved, the paper should say that SUDO targets the
same latent-scale parameter under the logit specification or provides a
link-flexible alternative. This affects the abstract, introduction, and
discussion.

### 3. The black-box inference claim exceeds the evidence

The paper states that a partially linear machine-learning imputation model
and a full-pipeline bootstrap attain nominal coverage. The repository
supports that statement in the black-box designs studied, with 100 outer
replications. It does not yet support a general validity claim.

The abstract and contributions should use design-specific language, such as:

> In the black-box designs studied, a tuned partially linear imputation model
> paired with a full-pipeline bootstrap attained nominal coverage.

The discussion already contains much of the necessary qualification.

### 4. The parametric theory is promising but not theorem-ready

The appendix appropriately describes its results as derivations with proof
sketches and numerical verification. The remaining gaps include:

- the contribution of the full-sample imputation estimate in the
  cross-fitted expansion;
- uniform control of stochastic remainders;
- conditional nuisance-rate requirements under bootstrap resampling;
- differentiability of the generated-outcome map;
- the correct fixed-imputation reference distribution;
- the flexible per-fold imputation regime.

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

### 5. Ordinal over-coverage is the primary empirical gap

The committed results confirm substantial conservatism:

- stage 4 reports coverage of 0.976;
- stage 5 at `n = 2000` reports SD 0.110, mean SE 0.149, and coverage 0.985;
- the internal Rubin checks report variance ratios of roughly 1.50 to 1.88.

This is much larger than the 1 to 2 percent Rubin-variance discrepancy
measured in the binary congenial testbed. The current evidence does not
justify attributing it to uncongeniality, finite-sample GAM behavior, fixed
imputations, or the threshold block alone.

The contrast already found in the repository is informative: ordinal
designs over-cover while binary designs with the same covariate structure
and nuisances under-cover. The ordinal effect also declines as signal grows.
This makes the threshold and category machinery a natural place to
investigate, but not a proven explanation.

The multiple-imputation literature supports the manuscript's caution.
Xie and Meng show that inference under uncongenial imputer and analyst
procedures has a complex dependence on their relationship and on
self-efficiency
([Statistica Sinica 27, 1485-1594](https://doi.org/10.5705/ss.2014.067)).
It does not support a generic claim that Rubin's variance safely overstates
uncertainty here.

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

### 8. The novelty claim needs a reproducible literature search

A broad search did not identify a direct competitor that combines ordinal
latent-scale outcomes, DML partialling, surrogate completion, and
multiple-imputation inference. That is encouraging but insufficient to prove
priority.

The internal novelty review should cover:

- DML for ordinal outcomes;
- debiased or orthogonal cumulative-link models;
- semiparametric partially linear transformation models;
- ordinal causal effects with machine-learned nuisances;
- latent-variable and distribution-regression DML.

Until this search is recorded, the paper should retain "to our knowledge"
and avoid unqualified priority claims.

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

### Phase 1: Isolate the ordinal variance anomaly

Status: the first isolation step is complete. The new
`R/theory_ordinal_variance_terms.R` stage includes the threshold block and
passes its Monte Carlo acceptance checks. It predicts `T/V = 1.028` at
moderate signal and `0.980` at strong signal, ruling out ordinal threshold
estimation and Rubin pooling alone as explanations for the stage-5 ratio of
1.50 to 1.88.

The remaining work is to add the nuisance components in a factorial ladder.

Remaining acceptance criteria:

- compare oracle and estimated linear nuisances;
- compare a fixed with a per-draw outcome nuisance;
- add GAM nuisances;
- vary sample size and imputation count;
- identify which transition reproduces the large stage-5 variance ratio.

This is the best first task because it is cheap, falsifiable, and central to
the ordinal inference claim.

### Phase 2: Repair assumptions and claims in the paper

Revise the manuscript without changing the method:

- formalize the conditional latent-error law;
- distinguish statistical identification from causal interpretation;
- replace "subsumes" unless estimator equivalence is proved;
- qualify black-box coverage claims by design;
- narrow the wine sensitivity conclusion;
- pair coverage claims with Monte Carlo uncertainty;
- explain the two stage-3p coverage fields, since the committed summary
  contains both 0.98 and 0.95.

Acceptance criterion: every statement in the abstract and contribution list
is supported by a theorem, a named simulation cell, or an explicit empirical
qualification.

### Phase 3: Complete the parametric asymptotic expansion

Treat the parametric full-sample imputation model before attempting the
cross-fitted black-box regime.

Required deliverables:

- a precise estimator definition including auxiliary randomness;
- an influence-function expansion for fixed imputation count;
- a lemma controlling the interaction of the full-sample imputation
  estimator with cross-fitted adjustment nuisances;
- uniform remainder conditions;
- explicit threshold-parameter contributions;
- a clear distinction between conditional-on-imputation and unconditional
  laws.

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
- what the bootstrap target is when the imputation count is fixed.

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

- test MARS as another partially linear full model;
- retune XGBoost using target-level Monte Carlo performance;
- measure the effective treatment-direction error directly across learners;
- record the systematic novelty search;
- leave multinomial outcomes as a separate project.

## Recommended execution order

The immediate sequence should be:

1. ordinal term-level decomposition;
2. paper-level assumption and claim repairs;
3. parametric asymptotic expansion;
4. full-pipeline bootstrap proof;
5. fixed-imputation reference law;
6. misspecification bound;
7. secondary learners and extensions.

This order gives the fastest opportunity either to validate the current
inferential account or to identify precisely where the estimator, variance,
or manuscript claims must change.
