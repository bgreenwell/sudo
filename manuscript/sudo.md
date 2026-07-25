# SUDO: Surrogate-Assisted Double Machine Learning

Working notes, restructured 2026-07 around the validated R ladder in `R/`
(stage numbers below refer to `R/stageN_*.R`; every numeric claim is
reproduced by the corresponding script and summarized in `R/results/`).

## 1. The problem: discrete outcomes break the PLR model

Double Machine Learning (Chernozhukov et al. 2018) estimates the causal
parameter of the partially linear regression (PLR) model

$$Y = \theta_0 D + g_0(X) + \varepsilon$$

by partialling out flexible, cross-fitted estimates of $E[Y|X]$ and $E[D|X]$
and regressing residual on residual. The score is Neyman-orthogonal, so
first-order errors in the ML nuisances do not contaminate $\hat\theta$.

The model is additive in a continuous outcome. When $Y$ is binary or ordinal
the additive structure lives on a latent scale, not the observed one:
applying the PLR machinery to discrete $Y$ either changes the estimand
(probability-scale contrast, no longer $\theta_0$) or is simply undefined.
The DoubleML libraries handle binary $Y$ only via a logistic partially
linear model (`DoubleMLLPLR`, Liu, Zhang & Zhou 2021) with the logit link
hardwired; nothing handles ordinal $Y$.

## 2. FWL: the backbone

The Frisch-Waugh-Lovell theorem is what makes partialling-out work: in a
linear model the coefficient on $D$ equals the coefficient from regressing
$Y$-residuals on $D$-residuals after both are partialled on $X$. DML is FWL
with ML nuisances plus cross-fitting.

Stage 0 verifies the ladder's foundation: residual-on-residual reproduces
the `lm` coefficient to machine precision (< 1e-14), and the cross-fitted
gam version of the PLR estimator is unbiased with nominal coverage
(bias +0.002, coverage 0.945 at n=1000, 200 reps), matching DoubleML-R on
the same data.

Everything that follows is one idea: **extend FWL to discrete outcomes by
completing the latent utility, then partial out as usual.**

## 3. Latent utility and surrogate completion

Assume the latent PLR (L-PLR) model

$$U = \theta_0 D + g_0(X) + \varepsilon, \qquad \varepsilon \sim F,$$

with observed $Y = \mathbf{1}\{U > 0\}$ (binary) or
$Y = j \iff c_{j-1} < U \le c_j$ (ordinal, thresholds
$c_0 = -\infty < c_1 < \dots < c_{J-1} < c_J = \infty$).

Following the surrogate-residual construction (Liu & Zhang 2018; Cheng,
Wang & Zhang 2021; implemented in the `sure` package, Greenwell et al.
2018), a surrogate $S$ is a draw from the conditional law of $U$ given the
observed category: sample from $F$ located at the estimated linear index
$\hat V$, truncated to the interval $Y$ implies. Binary is the $J=2$
special case with cutpoints $(-\infty, 0, \infty)$.

By Theorem 1 of Cheng et al. (2021), under correct specification
$E[S|X] = E[U|X]$, so substituting $S$ for $U$,

$$S = \theta_0 D + \ell_0(X) + \eta, \qquad E[\eta \mid D, X] = 0$$

holds exactly by construction — the equation FWL needs.

Stage 1 (binary, everything parametric): the surrogate FWL estimate, the
glm coefficient, and the oracle FWL on the true latent index all agree
(1.498 / 1.501 / 1.500 at $\theta_0 = 1.5$, n=5000, 200 reps).

**The full model must include D.** The imputation model estimates
$P(Y | D, X)$, not $P(Y | X)$: conditioning on $X$ alone biases $\hat V$
whenever $D$ explains $Y$ beyond $X$. The subsequent outcome nuisance
$E[S|X]$ is trained on $X$ only, stripping $D$ back out, and cross-fitting
prevents leakage. Stage 3(a) verifies both directions empirically.

## 4. Surrogates as multiple imputation (the organizing idea)

The latent utility is missing data; the full model of $Y$ on $(D, X)$ is
the imputation model; each surrogate draw is one completed dataset; the PLR
estimate on each completed dataset is one complete-data analysis. Rubin's
rules pool $B$ draws:

$$\bar\theta_B = \tfrac{1}{B}\sum_b \hat\theta^{(b)}, \qquad
T = \bar W_B + (1 + 1/B)\, B_B,$$

with $\bar W_B$ the mean within-draw (sandwich) variance and $B_B$ the
between-draw variance. Confidence intervals use the Barnard-Rubin (1999)
small-sample $t$ degrees of freedom. The mean, not the median, is required:
$B_B$ is defined by deviations from the mean.

Two consequences, both confirmed in stage 2 (n=5000, 500 reps):

1. **Single-draw inference under-covers badly** (0.804 at nominal 0.95):
   the sandwich SE sees one completed dataset and ignores draw uncertainty.
2. **Draws must be proper.** Redrawing $S$ around the same fitted index is
   improper imputation — it omits imputation-model parameter uncertainty
   and still under-covers (~0.93). Proper draws redraw the full-model
   parameters from their asymptotic posterior
   $N(\hat\beta, \widehat{\mathrm{Cov}}(\hat\beta))$ before each surrogate
   draw; coverage is then 0.96-0.97 for $B \in \{10, 25, 50\}$.

For ordinal models the thresholds are parameters too: proper draws perturb
$(\hat c, \hat\beta)$ jointly, which is why `ordinal::clm` (whose vcov
includes the thresholds) is the prototype full model — `mgcv::ocat` hides
threshold uncertainty from the covariance. Stage 4 confirms nominal
coverage with joint perturbation.

This resolves v1's open "under-coverage" problem: the missing variance was
the improper-imputation gap, not a defect of the partialling-out SE.

## 5. The DML layer: cross-fitting and orthogonality

With ML nuisances, the estimator is FWL on cross-fitted residuals:

$$\hat\theta_{\text{SUDO}} = \frac{\sum_i \tilde S_i \tilde D_i}{\sum_i \tilde D_i^2},
\qquad \tilde S_i = S_i - \hat\ell(X_i),\ \tilde D_i = D_i - \hat m(X_i).$$

**Neyman orthogonality.** The score is
$\psi(W; \theta, \eta) = (S - \ell(X) - \theta(D - m(X)))(D - m(X))$.
For the outcome nuisance $\ell$:
$\partial_\ell E[\psi]|_{\eta_0} = E[-(D - m_0(X))] = 0$ since
$E[D - m_0(X) | X] = 0$. For the treatment nuisance $m$, perturb
$m_t = m_0 + th$ and write $\tilde D_t = \tilde D_0 - th(X)$,
$\varepsilon = S - \ell_0(X) - \theta_0 \tilde D_0$:

$$\frac{d}{dt}\Big|_{t=0} E[\psi_t]
= E[\theta_0 h(X) \tilde D_0] + E[\varepsilon \cdot (-h(X))] = 0,$$

both terms vanishing by the tower property ($E[\tilde D_0 | X] = 0$ and
$E[\varepsilon | X] = 0$).

**Cross-fitting is required in the full model too.** A flexible classifier
trained on all data pushes fitted probabilities toward 0/1, compressing
$\mathrm{Var}(S)$ and biasing $\hat\ell$; all predictions are out-of-fold.

**Learner quality is a real constraint.** Stage 3 runs the full binary
pipeline (cross-fitted binomial gam full model + gam nuisances + proper MI)
across $\theta_0 \in \{0.5, 3\}$, $n \in \{1000, 2000\}$: unbiased
throughout, with nominal coverage at moderate $\theta_0$ (coverage at
$\theta_0 = 3$ degrades — §6 and the open problems explain why: the
pass-through factor is larger and the bootstrap under-propagates
full-model variance).
Rerunning v1's known-bias case ($\theta_0 = 3$, n=2000) with a
random-forest full model shows the rebuilt pipeline is unbiased there too
(+0.001) — v1's $\sim 0.9$ bias traced to its RF partialling nuisances, not
the surrogate mechanism. The RF arm does pay the improper-draw penalty
(no posterior to perturb: coverage 0.67), reinforcing that proper draws
carry the inference.

**Cheap draw design.** Nuisances are fit once; only $S$ (and the full-model
parameter draw) is redrawn across $b = 1..B$. Stage 3(b) shows refitting
$E[S|X]$ per draw changes nothing beyond MC error.

## 6. How full-model error propagates: the truncation pass-through

Section 5's orthogonality result covers the *partialling* nuisances
$\hat\ell, \hat m$. It says nothing about the full model, whose error is
baked into the constructed outcome $S$ rather than differenced away. This
section derives how that error propagates. It is the piece that explains
the black-box program's central puzzle — why *every* learner degrades as
the signal strengthens.

**The pass-through identity.** Conditional on the training folds, $\hat V$
is a deterministic function of $(D,X)$, while $Y \mid D,X \sim
\mathrm{Bern}(F(\eta))$ is generated by the *truth*, $F = \mathrm{plogis}$.
Since $S = \hat V + \varepsilon$ with $\varepsilon$ truncated to the side
$Y$ selects,

$$E[S \mid D, X] = \psi(\hat V, \eta), \qquad
\psi(v,e) = v + F(e)\,\mu_+(v) + (1 - F(e))\,\mu_-(v),$$

with $\mu_+(v) = E[\varepsilon \mid \varepsilon > -v]$ and
$\mu_-(v) = E[\varepsilon \mid \varepsilon \le -v]$. Because
$P(\varepsilon > -e) = F(e)$ by symmetry, $\psi(e,e) = e$ — which *is*
Theorem 1 of Cheng et al., recovered as a special case.

Differentiating in the first argument at $v = e$ (using
$\mu_+' = -f(v+\mu_+)/F$ and $\mu_-' = f(v+\mu_-)/(1-F)$) gives a clean
closed form for the **pass-through factor**:

$$c(e) \;=\; \frac{\partial \psi}{\partial v}\Big|_{v=e}
\;=\; 1 - f(e)\,\bigl(\mu_+(e) - \mu_-(e)\bigr), \qquad f = \mathrm{dlogis}.$$

| $e$ | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|---|---|
| $c(e)$ | 0.307 | 0.337 | 0.418 | 0.525 | 0.635 | 0.809 | 0.910 | 0.983 |

(verified against a numerical derivative of $\psi$ to $10^{-11}$.)

**Bias decomposition.** For binary $D$,
$E[h(D,X)(D-m)\mid X] = w(X)[h(1,X)-h(0,X)]$ with $w = m(1-m)$. Writing
the full-model error as $\hat V - \eta = \delta_D\,D + \delta_X(X)$ and
linearising $\psi$,

$$\mathrm{bias}(\hat\theta) \;\approx\;
\underbrace{\bar c_1 \cdot \delta_D}_{\text{$D$-direction}}
\;+\; \underbrace{\frac{E\bigl[w\,(c(\eta_1) - c(\eta_0))\,\delta_X\bigr]}{E[w]}}_{\text{$X$-leak}},
\qquad \bar c_1 = \frac{E[w\,c(\theta_0 + g)]}{E[w]}.$$

Under the stage-3 DGP: $\bar c_1 = 0.669$ at $\theta_0 = 1.5$ and
$\mathbf{0.861}$ at $\theta_0 = 3$; the $X$-leak multiplier is $+0.192$
and $+0.384$.

**Three consequences.**

1. **The truncation self-corrects, but only partially, and less as the
   signal grows.** $Y$ is generated by the truth, so it drags $S$ back
   toward the correct interval — that is why $c < 1$. But $c(e) \to 1$ as
   $|e|$ grows: once $Y$ is nearly deterministic it carries no information
   beyond what $\hat V$ already says, and the correction switches off.
   **This, not quasi-separation, is why every learner is worse at
   $\theta_0 = 3$.**
2. **The $X$-direction is not protected either**, contrary to a natural
   first guess. The tempting argument — $E[(D-m)h(X)] = 0$ annihilates any
   $X$-only error — fails because $S$ is a *nonlinear* function of $\hat V$,
   so the pass-through $c(\eta)$ itself depends on $D$. The leak is
   genuinely first order; it is merely small here, and it doubles from
   $\theta_0 = 1.5$ to $3$.
3. **SUDO is structurally more fragile than standard DML.** In DML the
   outcome is observed and nuisance error enters only through a *product*
   of two errors. SUDO adds a third nuisance whose error enters
   **linearly in both directions**, with multipliers computable in closed
   form from the assumed link.

**Empirical support.** Modelling each learner as a multiplicative
attenuation of the effective $D$-coefficient, $\delta_D = (a-1)\theta_0$,
and solving $a$ from the committed biases, a *single* $a$ per learner
explains *both* signal strengths:

| learner | $a$ @ $\theta_0{=}1.5$ | $a$ @ $\theta_0{=}3$ |
|---|---|---|
| PL mboost | 0.834 | 0.829 |
| PL backfit (untuned) | 0.977 | 0.959 |
| PL backfit (3m-tuned) | 0.973 | 0.989 |
| PL xgboost | 0.906 | 0.938 |

**Where the linearisation breaks.** Stage 3h's re-analysis
(`R/stage3h_reanalysis.R`) compares two routes to $\delta_D$: the exact
identity $\delta_D = \mathrm{bias} - \text{(D-projection of } S - \hat V)$,
which follows from linearity of FWL, against the theory route
$\delta_D = \mathrm{bias}/\bar c_1$. They agree to $\le 0.027$ for `gam`,
tuned `nnet`, and `ranger` at $\theta_0=1.5$, but diverge by $0.06$-$0.08$
for the untuned net and for `ranger` at $\theta_0 = 3$ — precisely the
learners whose indices are worst fit ($R^2 \approx 0.41$-$0.66$) and most
often clamped. All eight gaps have the *same sign*, consistent with the
positive $X$-leak term the decomposition predicts. So the pass-through
model is a good account of well-behaved full models and an incomplete one
for badly-fitting indices — which is exactly why $\delta_D$ must be
**measured directly** rather than inferred from the bias.

## 7. Ordinal outcomes

The cumulative-link model supplies the estimated linear index and
thresholds; the surrogate is drawn from $F$ located at $\hat V_i$ truncated
to $(\hat c_{Y_i - 1}, \hat c_{Y_i}]$; the downstream FWL + Rubin machinery
is unchanged. This is the case with no standard DML competitor at all.

- Stage 4 (parametric, `clm`, J=3): bias +0.002, coverage 0.976 at n=3000.
- Stage 5 (flexible: `clm` on natural-spline bases, fit once — its vcov
  still covers the thresholds, so draws stay proper — with cross-fitted gam
  partialling nuisances): bias +0.015/+0.022 at n=1000/2000 (a ~2%
  spline-approximation bias that trades off against small-sample behavior:
  df=6 doubles the n=1000 bias), coverage 0.99 — conservative, see below.
  Cross-fitting the full model is wrong here: per-fold index noise inflates
  the between-draw variance (coverage 1.0 with mean-se at 2.3x the true sd);
  parametric full models are fit once, the stage-2/4 design.

Ordinal outcomes carry more threshold information per observation than
binary, which sharpens both estimation and diagnostics (stage 6: the same
diagnostic that needs n≈20000 in the binary case fires at n=5000 ordinal).

Multinomial (nominal) outcomes are future work: the surrogate is a
J-vector drawn by Gibbs sampling over truncated multivariate-Gumbel
conditionals, there is no scalar latent effect, and the surrogate is
diagnostic-only (v1 SIM2 demonstrated the diagnostic).

## 8. Link misspecification and diagnostics

$F$ is an assumption. Stage 6, with Gumbel errors and a logistic analyst:

- wrong link: bias +0.77 (binary), +0.58 (ordinal), coverage 0
- right link (cloglog completion): bias < 0.005, nominal coverage

Sign conventions matter: binary `glm` cloglog implies a Gumbel-max latent
error, while `ordinal::clm` cloglog implies Gumbel-min
(`complete_surrogate` exposes `cloglog` and `cloglog_min`).

**Diagnosing the link.** Surrogate residuals $R = S - \hat V$ follow the
assumed law and are homoscedastic under correct specification (Cheng et
al. 2021). Conditional-MEAN tests are nearly blind to pure link
misspecification — flexible thresholds absorb the misfit — and marginal KS
tests need huge n. The sharp signal is **variance drift**: regress $R^2$
on a smooth of $X$ (`sure::resids` + `mgcv`); under the wrong link the
smooth is significant at n=5000 (p ≈ 2e-4) while the right link is clean
(p ≈ 0.4).

**Independent draw streams.** Diagnosis uses its own surrogate draws on a
separate RNG stream, never the estimation draws — resolving v1's
shared-draw dependence concern.

## 9. The SUDO algorithm

1. **Full (imputation) model.** Cross-fit a model of $Y$ on $(D, X)$ over
   $K$ folds; record out-of-fold linear index $\hat V_i$, thresholds
   $\hat c$, and each fold's parameter covariance.
2. **Treatment and outcome nuisances.** Cross-fit $\hat m(X) \approx E[D|X]$;
   draw an initial surrogate and cross-fit $\hat\ell(X) \approx E[S|X]$
   (on $X$ only). Compute $\tilde D_i$.
3. **Proper surrogate draws.** For $b = 1..B$ (B = 25-50): redraw the
   full-model parameters from $N(\hat\beta, \widehat{\mathrm{Cov}})$
   fold-wise, recompute the index and thresholds, draw $S^{(b)}$ from the
   truncated link law, form $\tilde S^{(b)} = S^{(b)} - \hat\ell(X)$, and
   compute $\hat\theta^{(b)}$ and its sandwich variance by FWL.
4. **Pool.** Rubin's rules with Barnard-Rubin df give $\bar\theta_B$, total
   variance $T$, and the CI.
5. **Diagnose.** On an independent draw stream, check surrogate-residual
   variance drift in $X$ (and QQ against the assumed law); if flagged,
   respecify the link and rerun.

## 10. Estimand

$\theta_0$ is the treatment effect on the **latent utility scale** (log-odds
scale under logit) — the same estimand as a correctly specified parametric
cumulative-link model, freed from parametric $g(X)$. It is not the
probability-scale ATE; naive DML on binary $Y$ targets the latter, so the
two are not comparable against the same truth. Probability-scale summaries
can be derived post hoc from the latent model when needed.

## 11. Open problems

- Formal proof that including $D$ in the full model leaves the score
  orthogonal (stage 3(a) verifies empirically).
- **Under-coverage at extreme signal is now localised (stages 3n/3o).**
  It was originally attributed to quasi-separation of the gam full model.
  §6 gives the deeper reason — the pass-through factor $\bar c_1$ rises
  from 0.669 to 0.861, so full-model error does more damage — and the
  oracle-index control (stage 3n) pins the *variance* half down exactly:
  (a) given the true index, the Rubin/bootstrap SE is nearly twice too
  large ($\mathrm{sd}/\mathrm{mean\ se} = 0.48$, coverage 1.000) — the
  pooling machinery *over*-covers, so it is not the culprit; (b) estimating
  the partialling nuisances with gams costs nothing
  ($0.48 \to 0.48$) — orthogonality holds; (c) a *fitted* index quadruples
  the true sampling SD ($0.057 \to 0.20$ at $\theta_0 = 3$) and flips
  $\mathrm{sd}/\mathrm{mean\ se}$ to $1.25$-$1.29$. The shortfall is SE
  *level* not *variability* (a $1.96\,\overline{\mathrm{se}}$ interval
  covers the same as the reported CI), which rules out a noisy $T$ from
  small $B$. Diagnosis: **the recentered bootstrap under-propagates the
  full model's own sampling variance into $B_B$.** Candidate fixes,
  untried: a full-pipeline bootstrap (resample before cross-fitting, not
  just refit the full model), or a parametric proper draw for the PL
  model's $\hat\beta$ (which has an asymptotic covariance).
- **First-order vs second-order sensitivity, measured (stage 3o).** With
  the outcome observed, DML is unbiased under an *arbitrarily* wrong
  outcome nuisance as long as the propensity is right (product-of-errors
  orthogonality); with both nuisances wrong the bias is quadratic. SUDO's
  imputation-model error enters *linearly*, and the $D$-direction slope is
  the pass-through constant $\bar c_1$ — measured at 0.669 / 0.861,
  matching the §6 derivation to $\le 0.001$. This is the genuine
  structural cost of the surrogate approach and belongs in the paper's
  discussion as such.
- **Black-box full models (stages 3d-3g).** Five attempts, in order.
  (i) *Bootstrap refits* as the proper-draw substitute for an RF full
  model: raw refits are location-shifted (a forest refit on a resample
  smooths harder) and bias $\theta$ (+0.125 at $\theta_0 = 3$); recentering
  at the original fit removes that and lifts coverage 0.665 $\to$ 0.765 —
  short of nominal. (ii) The recentering experiment's control arm found a
  second, independent problem: the RF index attenuates $\theta$ at moderate
  signal ($-0.16$ at $\theta_0 = 1.5$, identical under improper and
  bootstrap draws) — RF probabilities compress toward the base rate, an
  effect invisible at strong signal. *Isotonic recalibration* of the RF
  probabilities (the standard fix for miscalibration) barely dents this and
  badly breaks the strong-signal case (bias $-0.34$ at $\theta_0 = 3$): it
  pools the near-separated tail probabilities into ties, flattening the
  variation strong-signal estimation depends on. (iii) Hypothesis: the
  attenuation is specific to piecewise-constant learners; a smooth net
  predicting the log-odds directly should avoid it. An *untuned* net
  (`decay=0.01`) produced a badly biased $\theta$ (+0.43 to +0.47 at
  $\theta_0=3$, worse than RF) — before reporting "smooth learners don't
  help either," an in-sample and cross-fitted decay/size grid traced this
  to under-regularization, and `decay=0.1` (size 8) removed nearly all of
  it. (iv) With that tuning, the net nearly eliminates the moderate-signal
  attenuation (bias $-0.169\to-0.010$ improper, $-0.160\to+0.009$
  bootstrap) and reaches coverage 0.917 at $\theta_0=3$ under bootstrap
  draws — about one MC-SE from nominal, the best result of the program.
  (v) A direct diagnostic (stage 3g): with the true systematic index
  $\eta = \theta_0 D + g(X)$ known in simulation, regress $\eta$ on each
  learner's cross-fitted $\hat V$ (no surrogate draws, no FWL). Note the
  direction: the regression is $\eta$ on $\hat V$, so slope $>1$ means
  $\hat V$ is *attenuated* and slope $<1$ means it is *inflated*. The
  mapping from this marginal calibration to $\theta$-scale bias is not
  monotonic. Concretely, at $\theta_0 = 3$,
  `ranger` has the worst index fit of anything tested (slope $0.57$,
  $R^2 = 0.41$) yet the best $\theta$ point estimate, while the untuned
  net has a *better* index fit (slope $0.65$, $R^2 = 0.63$) yet the worst
  $\theta$ bias. `gam` remains essentially calibrated (slope
  $\approx 1$, $R^2 \ge 0.91$) at both signal strengths, consistent with
  its reliable $\theta$ behavior throughout. Practical upshot: index-scale
  calibration is a useful sanity check (a well-calibrated full model is
  trustworthy) but not a substitute for full $\theta$-level MC validation
  of a black-box learner — there is no shortcut diagnostic yet.
  (vi) *A sharper, $D$-specific diagnostic, also negative* (stage 3h):
  since $\theta_0$ is the coefficient on $D$ specifically, draw a
  surrogate residual $R = S - \hat V$ and fit `gam(R \sim D + s(X))`; a
  trustworthy full model should show coefficient $\approx 0$ (no
  $D$-specific structure in the residual). It does not track $\theta$
  bias either: at $\theta_0 = 3$, `ranger` (best $\theta$ estimate of any
  black-box learner) has the *largest* mean $|{\rm coef}|$ (0.074) of the
  four learners tested, while the untuned net (worst $\theta$ bias,
  +0.457) has one (0.067) barely distinguishable from `gam`'s (0.067);
  within-rep correlations between $|{\rm coef}|$ and $|\theta$ bias$|$ are
  all small and inconsistent in sign ($-0.01$ to $+0.10$). One number did
  move as expected: the untuned net's residual variance ratio between
  treatment arms (0.89 at $\theta_0=3$) is the furthest from 1 of any
  learner, hinting the problem may live in treatment-conditional variance
  rather than mean structure.
  (vii) *Testing the variance-ratio lead directly, with partial success*
  (stage 3i): a proper hypothesis test (`var.test()` on the surrogate
  residual split by $D$, not just the raw ratio), same seeds as (vi) for
  dataset-by-dataset comparability. This is the first diagnostic in the
  program to correctly separate the pathological learner from the
  trustworthy ones with real statistical confidence: the untuned net's
  variance ratio departs from 1 with overwhelming significance
  ($p = 2.6\times10^{-13}$ at $\theta_0=1.5$, $p = 1.7\times10^{-27}$ at
  $\theta_0=3$), while `gam` and `ranger` are never significant
  ($p \in [0.09, 0.68]$), and the cross-learner ranking by
  $|{\rm ratio}-1|$ gets both extremes right (untuned net worst, `gam`
  best) at both signal strengths. It is not a full solution: `ranger` and
  the tuned net swap order in the middle of the ranking, and — unlike the
  aggregate signal — the within-rep correlation between $|{\rm ratio}-1|$
  and $|\theta$ bias$|$ on the *same* dataset stays near 0 everywhere.
  Practical reading: useful for *learner selection* (a repeated-sampling
  check when choosing a candidate full model or tuning hyperparameters —
  exactly the round-2 use case) but not for judging whether one
  already-fitted result on one real dataset is trustworthy. Three
  diagnostics tried total: two negative (index calibration, $D$-residual
  mean structure), one partially positive (treatment-conditional
  variance) — **there is still no known per-dataset diagnostic for
  black-box full-model trustworthiness; full $\theta$-level MC validation
  remains not optional for judging any single fitted result**, though the
  variance test is a real candidate for the learner-selection step of a
  future tuning pipeline. Still open: improper-draw coverage trails
  proper inference everywhere (structurally, since no scheme here has a
  true posterior); the NN tuning was manual probing, not a principled
  search — a cross-validated tuning step belongs inside the pipeline, and
  should select hyperparameters by $\theta$-level MC performance
  directly (the variance test as a screening step, not a replacement);
  gradient boosting and a full-pipeline (not just full-model) bootstrap
  remain untried in the non-PL setting.
  (viii) *Partially-linear (PL) full models, three ways* (stages 3j-3l):
  $\hat V = \hat\beta D + \hat f(X)$ with $D$ structurally forbidden from
  interacting with $X$, tested despite the mean-structure test (vi) not
  having confirmed $D\times X$ interaction as the mechanism — a
  structural safeguard rather than a diagnosed fix. *Backfitting*
  (IRLS/profile-likelihood, $\hat f$ via the untuned `nnet` that produced
  the program's worst result) is the best outcome of the whole program:
  with no tuning search, bootstrap coverage reaches 0.91 at
  $\theta_0=1.5$ (beating the manually-tuned net's 0.883) and 0.83 at
  $\theta_0=3$ (vs. the tuned net's 0.917) — closing most of the gap that
  previously required a regularization search. *XGBoost* with
  `interaction_constraints` (the constraint verified by tree-dump
  inspection: 0 of 50 trees combine a $D$-split with an $X$-split) helps
  only modestly (coverage 0.85/0.73, comparable to plain `ranger`).
  *mboost* componentwise boosting (`bols(D) + bbs(X_j)`, `mstop` via
  `cvrisk()`) is an outright failure — coverage 0.70/0.11, the single
  worst result in the entire program, despite the theoretically purest PL
  construction of the three (literally no interaction basis function can
  exist). Its calibration diagnostic is consistent with this: slope
  1.31-1.34 means the index is *attenuated* by a factor $\approx 1.3$
  (the regression is $\eta$ on $\hat V$), matching its heavily attenuated
  $\theta$ — mboost is in fact the most internally consistent learner in
  the table, not an exception. It achieves this alongside the best index
  $R^2$ of any learner tested (0.92-0.93, above `gam`'s), which is the
  point: `cvrisk()` optimises predictive deviance, not $D$-coefficient
  calibration, and componentwise boosting's greedy stagewise selection
  gives no guarantee that early stopping leaves the `bols(D)` term
  correctly scaled against the smooths. Excellent overall fit and a badly
  scaled treatment coefficient are entirely compatible.
  (ix) *This overturns finding (vii)'s "variance test useful for learner
  selection" conclusion.* Run on the three PL models, the
  variance-ratio diagnostic dissociates from $\theta$ bias in *both*
  directions: backfitting's ratio (0.90-0.96, still highly significant,
  $p$ as low as $2.8\times10^{-27}$) is nearly as extreme as the
  pathological untuned net's, despite backfitting fixing most of the
  actual bias; mboost's ratio (0.97-0.99, barely/not significant) looks
  clean despite being the worst result in the program. Across seven
  learners now tested, no diagnostic tried — index calibration, $D$
  -residual mean structure, or treatment-conditional variance — reliably
  predicts $\theta$ bias for an arbitrary black-box or semi-structured
  full model; only the actual bias/coverage MC does. Practical upshot:
  backfitting is the best-known base case for a future CV'd tuning
  search, but the search criterion must be $\theta$-level performance
  itself, since mboost shows a "principled" CV procedure optimizing the
  wrong objective can be actively harmful. `earth`/MARS with `linpreds`
  (forcing $D$ linear, $X$ MARS-adaptive) is a fourth PL route, untried.
- Asymptotics for proper-MI Rubin inference here: the draws share one
  dataset, and proper draws inject the full model's own $\beta_D$ posterior
  into $T$, which can exceed the DML sampling variance — Rubin under
  uncongeniality (Meng 1994) errs conservative (stage 4: +0.03; stage 5
  with a rich basis: up to +0.05), never anti-conservative in our runs.
- Multinomial extension (Gibbs surrogate vector, SUR partialling).
- Python parity for cloglog ordinal (statsmodels `OrderedModel` lacks
  cloglog; cumulative-binary decomposition is the workaround).
- Real-data application: Plan B replication (`archive/simulations-v1/planb_replication.py`)
  to be rebuilt on the new package.

## Key references

- Chernozhukov et al. (2018). Double/debiased machine learning. *Econometrics J.*
- Liu, Zhang & Zhou (2021). Double machine learning for logistic partially
  linear models. arXiv:2009.14461.
- Liu & Zhang (2018). Surrogate residuals for ordinal regression. *JASA*.
- Cheng, Wang & Zhang (2021). Surrogate residuals for general models. 
- Greenwell, McCarthy, Boehmke & Liu (2018). Residuals and diagnostics for
  ordinal regression models: the sure package. *R Journal*.
- Barnard & Rubin (1999). Small-sample degrees of freedom with multiple
  imputation. *Biometrika*.
