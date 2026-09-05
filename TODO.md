# SUDO TODO

## 1. Verify agreement with the surrogate literature and `sure`

- [x] Add a standalone R validation stage, run from the repository root.
- [x] Determine when randomized SUDO uses the same latent surrogate-response distribution as `sure::surrogate(method = "latent")`, allowing for equivalent sign, location, and threshold conventions, and document the binary cloglog exception.
- [x] Verify the comparison for binary and ordinal outcomes under the supported links. Stage 20 finds exact agreement for binary logit/probit and ordinal logit/probit/cloglog after the threshold-origin shift. The local `sure` development version uses the ordinal Gumbel-min law for binary cloglog, so it does not match the Gumbel-max law required by a binomial cloglog GLM.
- [ ] Show that the analytic Rao-Blackwell surrogate response equals the Monte Carlo average of many randomized surrogate responses.
- [x] Reproduce the `sure` diagnostic residual as
  `S - E(S | D, X)`.
- [ ] Demonstrate why SUDO instead residualizes against `X` alone,
  `S - E(S | X)`, so the treatment signal is retained.
- [x] Include explicit numerical assertions and fail the stage when any equivalence check exceeds its stated tolerance.

## 2. Freeze the paper's contribution statement

- [x] State clearly that SUDO does not introduce the latent surrogate response.
- [x] Credit the surrogate-residual literature and the `sure` implementation for that construction.
- [x] Define SUDO's contribution as using the surrogate response as an outcome for treatment-effect estimation.
- [x] Explain that the full outcome model includes treatment, while the DML nuisance removes only the covariate component.
- [x] Describe cross-fitting, proper multiple imputation, Rubin pooling, and the full-pipeline bootstrap in plain language.
- [x] Emphasize that the target is the treatment effect on the latent-utility scale and that black-box full models require target-level Monte Carlo validation.

## 3. Run a fresh ordinal confirmation experiment

- [ ] Preserve the completed Stage 19 results as the exploratory record.
- [ ] Create a new standalone confirmation stage with fresh seeds and a frozen learner specification.
- [ ] Use the selected `mboost::PropOdds` specification with `mstop = 2000`; do not retune on the confirmation simulations.
- [ ] Make SUDO bias and interval coverage the primary acceptance criteria.
- [ ] Report the direct proportional-odds treatment coefficient as a diagnostic, not as a gate for SUDO.
- [ ] Include the oracle latent partially linear estimator as a positive control.
- [ ] Include an ordinal model with the true covariate function supplied as an offset to isolate treatment-coefficient behavior.
- [ ] Target at least 100 Monte Carlo samples with 99 full-pipeline bootstrap replicates, using resumable output and macOS-safe PSOCK parallelism.
- [ ] Report bias relative to Monte Carlo standard error, empirical standard deviation, average estimated standard error, and coverage with its Monte Carlo uncertainty.

## 4. Explain the mechanism

- [ ] Add a focused experiment showing how SUDO can recover the latent treatment effect even when the direct ordinal model's treatment coefficient is attenuated.
- [ ] Compare the fitted ordinal index, the randomized surrogate response, the Rao-Blackwell surrogate response, and the final residualized estimating equation.
- [ ] Separate errors in the full outcome model from errors in the DML partialling nuisances.
- [ ] Use the experiment to determine which explanation belongs in the main paper and which diagnostics belong in the appendix.

## 5. Draft the core paper sections

- [ ] Revise the introduction around the latent-scale estimand and the failure of arbitrary numeric category scores.
- [ ] Finalize the setup and estimand section.
- [ ] Finalize the relationship-to-surrogate-literature section using the terminology of the cited papers and `sure`.
- [ ] Finalize the SUDO algorithm, making the generation of the surrogate response an explicit step.
- [ ] Add the fresh ordinal confirmation only after its R stage passes all SUDO acceptance criteria.

## Immediate next action

- [x] Implement and run the `sure` equivalence stage before expanding the paper prose or porting any ordinal method changes to Python.
