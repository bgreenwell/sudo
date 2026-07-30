# Review of the SUDO manuscript and research agenda

> **Status: resolved (2026-07-30).**
>
> All eight major findings have been addressed in the manuscript. This review
> is retained as a historical record. See
> [open_problems.md](../open_problems.md) for the live research agenda.

## Scope

This review covers the paper and appendices in
`manuscript/paper/sudo_paper.qmd`, the active agenda in
`manuscript/open_problems.md`, the committed simulation summaries, and the
earlier external assessment that initiated several of the theory checks. It
evaluates the manuscript against the repository evidence and the primary
sources cited in the paper.

## Overall assessment

SUDO has a credible methodological core, an original use of surrogate
residuals as latent-outcome completions, and an unusually auditable
validation record. The pass-through analysis explains why the full
imputation model matters at first order, the ordinal correction is isolated
rather than patched empirically, and the fixed-imputation analysis states
clearly where Rubin inference is and is not exact.

The paper now separates three levels of evidence:

1. Propositions A1 to A4 derive the pass-through map, moment properties, and
   misspecification bounds directly.
2. Theorems S1 and S2 prove the projected expansion and full-pipeline
   bootstrap result for a deterministic logistic-series reference learner.
3. Implications M1 and M2 state the corresponding adaptive-learner result.
   GAM, MARS, and neural backfitting satisfy the numerical criteria in the
   designs studied, but their projected asymptotic linearity and bootstrap
   stability remain unproved.

That boundary is the manuscript's principal theoretical limitation. It is
specific and does not invalidate the parametric results or the proved series
specialization.

## Major findings

### 1. The novelty claim must be estimand-specific

Ordinary DML methods can estimate probability- and distribution-scale
effects for binary or ordinal outcomes. The gap addressed by SUDO is
narrower and more precise: a partially linear latent-utility coefficient for
binary and ordinal outcomes, with link-flexible surrogate completion.

The manuscript should therefore not say that categorical outcomes have no
DML methods generally. Its defensible claim is that the logistic partially
linear estimator of Liu, Zhang, and Zhou targets the same binary latent-logit
coefficient under the logit specification, while SUDO adds other links and
an ordinal latent-scale counterpart. The paper makes no estimator-equivalence
or common-efficiency claim.

### 2. The identification assumptions are explicit

The completion identity requires the conditional latent-error law, not only
a marginal statement about the error distribution. The setup now states
$$
\varepsilon\mid D,X\sim F
$$
with the appropriate scale and threshold normalizations. It separates
latent ignorability, the conditional location-family model, overlap, and the
constant latent-scale effect.

This condition is foundational. The identity
$E[S\mid D,X]=E[U\mid D,X]$, the link diagnostic, and the pass-through
derivatives all depend on it.

### 3. The adaptive bootstrap claim is numerical, not yet general

The full-pipeline bootstrap captures uncertainty that a within-fold refit or
an improper fixed-index procedure misses. Across GAM, MARS, and neural
backfitting, the fixed/redrawn mean-SE ratio is 0.97 to 1.08. The within-fold
SE is 0.90 to 0.93 of the full-pipeline SE at moderate signal and 0.75 to
0.77 at strong signal. Normal coverage is 0.93 to 0.97 in the paired
bootstrap study.

Implication M2 supports a normal interval based on the full-pipeline bootstrap
standard deviation if its high-level conditions hold. It does not justify a
conventional percentile or studentized interval at fixed imputation count,
because fresh surrogate randomness leaves an order-$n^{-1/2}$ centering gap.
Those intervals remain empirical comparisons.

Theorem S2 proves the required bootstrap stability for the deterministic
series reference. The remaining obligation is to verify the projected
condition, uniform remainder, and conditional nuisance rates separately for
the three adaptive algorithms. Tang and Westling provide general conditions
for bootstrapping asymptotically linear estimators with machine-learned
nuisances
([arXiv:2404.03064](https://arxiv.org/abs/2404.03064)). Lin and Han establish
exchangeably weighted bootstrap validity for general DML estimators under
the conditions needed for DML itself
([arXiv:2604.17239](https://arxiv.org/abs/2604.17239)). Neither result
automatically supplies SUDO's projected learner expansion or its
generated-outcome continuity condition.

### 4. Partial linearity is a restriction, not a proof

Constraining the imputation index to
$$
\hat v(D,X)=\hat\alpha D+\hat f(X)
$$
prevents treatment-by-covariate artifacts. It does not guarantee that
$\hat f$ consistently estimates the covariate component, that
$\hat\alpha$ has the required projected expansion, or that the bootstrap
version is stable.

The common comparison nevertheless gives useful design-specific evidence.
GAM, MARS, and neural backfitting all pass the original Monte-Carlo-aware
bias and coverage criteria. MARS is correctly implemented by constructing
bases from $X$ only and then estimating a mandatory linear treatment term in
a joint logistic fit. `earth::linpreds` alone would still permit treatment
interactions.

The deterministic Hermite series serves a different purpose. It is a proof
reference, not the recommended adaptive learner. Theorem S1 derives its
projected influence expansion, and Theorem S2 establishes bootstrap
stability under explicit series and conditional bootstrap conditions.

### 5. The ordinal variance anomaly is resolved

The original ordinal implementation substantially over-covered. A
term-level decomposition including threshold uncertainty showed that
threshold estimation and Rubin pooling alone could not explain the variance
gap. The paired nuisance ladder then isolated reuse of one plug-in estimate
of $E[S\mid X]$ across completions.

Refitting the outcome nuisance on every ordinal completion leaves empirical
point-estimator variance nearly unchanged but reduces the within-draw
component. In the paired ladder, $T/V$ falls from 1.505 to 0.942. The full
corrected stage at $n=2000$, $B=25$, and 200 replications reports bias 0.016,
coverage 0.960, and $T/V=1.214$ with Monte Carlo standard error 0.122. It
passes the prespecified checks and is now the R and Python default.

The paper should describe the old implementation as the experiment that
exposed the problem, not as a remaining anomaly.

### 6. Fixed-imputation inference has a reference law

When $B$ is fixed, the scaled between-imputation variance retains a
chi-square fluctuation as $n$ grows. The limiting statistic is a normal
variable divided by the square root of a constant plus a scaled chi-square
variable, not generally a Student distribution.

At strong signal, predicted normal coverage is 0.888, 0.921, and 0.938 for
$B=5,10,25$. Barnard-Rubin gives 0.942, 0.947, and 0.948, while the
variance-calibrated Satterthwaite approximation remains within 0.006 of
0.95. Prediction and finite-sample coverage differ by at most 0.012 across
the validation grid. Barnard-Rubin with at least 25 imputations remains the
practical parametric default because it needs no additional component
estimators.

Increasing $B$ removes Monte Carlo uncertainty in the between-imputation
sample variance. It does not imply that every imputation-related variance
component disappears.

### 7. Rubin exactness remains a narrow open question

Conjecture A2 shows that Rubin's variance is asymptotically exact precisely
when
$$
G'C=\sigma_u^2.
$$
The identity fails in both directions in the congenial testbed, producing
about 1% to 2% variance discrepancies. The open question is whether a useful
subclass forces the identity or bounds its failure. There is no general
one-sided conservatism result here.

This discrepancy is theoretically important but practically smaller than
the fixed-$B$ chi-square variation at ordinary imputation counts.

### 8. The wine comparator needs neutral interpretation

The cumulative-link coefficient is not a naive estimate that simply ignores
completion uncertainty. It is a model-based estimator that can be efficient
when the full parametric outcome model is correct. Its narrower standard
error reflects stronger functional-form assumptions.

SUDO uses that model to construct completions but leaves the partialling
regressions flexible and propagates completion and cross-fitting variation.
The application should present the cumulative-link coefficient as a
stronger-model comparator rather than evidence of under-propagated
uncertainty.

The causal interpretation remains conditional on a debatable graph and
strong adjustment assumptions. The treatment-direction sensitivity
calculation is also conditional on negligible covariate-direction
misspecification. Proposition A4 supplies the exact covariate-leak identity
and weighted bound for a specified perturbation; it does not establish
general robustness.

## Assessment of the earlier external review

The earlier assessment was useful as an agenda check but not reliable as a
technical review. Its strongest suggestions were independently investigated:

- full-pipeline bootstrap validity;
- the generated-outcome step as distinct from ordinary DML;
- an ordinal term-level variance decomposition;
- threshold contributions to the derivative;
- fixed-imputation reference distributions;
- theory for cross-fitted flexible imputation models;
- MARS as a partially-linear learner.

Several claims should not be retained:

1. Its bootstrap checklist was plausible, not an established theorem for
   SUDO.
2. Redrawing folds is not known to be necessary. Fixed and redrawn regular
   folds are both admissible under Implication M2 and similar numerically.
3. Threshold derivatives alone do not solve the ordinal variance problem.
4. Firth or ridge estimation is not known to be mandatory.
5. The between-imputation variance component does not necessarily vanish as
   $B$ grows; its sample-variance noise does.
6. Nonsmooth trees were not shown to cause XGBoost's failure.
7. `earth::linpreds` does not by itself enforce the required MARS structure.
8. Weak secondary sources and broken mathematical image references cannot
   support theoretical conclusions.
