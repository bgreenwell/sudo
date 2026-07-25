# Stage 3h re-analysis: the original summary reported mean(|coef|), which
# destroyed the signal it was built to detect.
#
# Stage 3h asked whether the full model's surrogate residual R = S - V_hat
# carries D-specific structure that predicts theta bias, via
# gam(R ~ D + s(X)). It was written up as NEGATIVE. That verdict is an
# absolute-value artifact: the D-coefficient is a SIGNED quantity whose
# expected value under the pass-through theory is
#     E[resid_D_coef] = (cbar_1 - 1) * delta_D,
# i.e. negative-times-negative for an attenuating learner. Averaging
# |coef| over reps replaces a mean near +-0.07 with a folded-normal mean
# near 0.06-0.08 for EVERY learner, which is exactly the "all learners
# look alike" non-result that was reported.
#
# This script re-reads the committed per-rep detail (no MC re-run) and
# reports the signed statistics.
#
# CAVEAT, and the reason this is a re-analysis rather than a rehabilitation:
# stage 3h computed R and theta_hat from the SAME surrogate draw, which
# violates the repo's own independent-draw-stream convention (CLAUDE.md).
# Part of the within-rep correlation below is therefore shared draw noise,
# not diagnostic signal. A clean re-run on independent streams is required
# before this diagnostic can be called positive.
#
# Run from repo root: Rscript R/stage3h_reanalysis.R

d <- read.csv("R/results/stage3h_detail.csv")

rows <- list()
cat("stage 3h re-analysed with SIGNED coefficients (n_reps = 100 per cell)\n")
cat("theory: E[resid_D_coef] = (cbar_1 - 1) * delta_D; cbar_1 = 0.669 (t1.5), 0.861 (t3)\n\n")
cat(sprintf("%5s %-11s %18s %10s %12s %10s\n",
            "theta", "learner", "mean SIGNED (se)", "mean |.|",
            "mean bias", "cor(signed)"))
for (th in sort(unique(d$theta))) {
  for (L in unique(d$learner)) {
    s <- d[d$theta == th & d$learner == L, ]
    m <- mean(s$resid_D_coef)
    se <- sd(s$resid_D_coef) / sqrt(nrow(s))
    cr <- cor(s$resid_D_coef, s$theta_bias)
    cat(sprintf("%5.1f %-11s   %+.4f (%.4f) %10.4f %12.4f %10.3f\n",
                th, L, m, se, mean(abs(s$resid_D_coef)),
                mean(s$theta_bias), cr))
    rows[[length(rows) + 1]] <- data.frame(
      stage = "3h-reanalysis", theta = th, learner = L, n_reps = nrow(s),
      mean_signed_resid_D = m, se_signed_resid_D = se,
      mean_abs_resid_D = mean(abs(s$resid_D_coef)),
      t_stat_vs_zero = m / se,
      mean_theta_bias = mean(s$theta_bias),
      cor_signed = cr,
      cor_abs = cor(abs(s$resid_D_coef), abs(s$theta_bias)))
  }
}
sm <- do.call(rbind, rows)

cat("\n== what changed ==\n")
sig <- abs(sm$t_stat_vs_zero) > 3
cat(sprintf("cells with |t| > 3 for the SIGNED coefficient: %d of %d\n",
            sum(sig), nrow(sm)))
cat(sprintf("  %s\n", paste(sprintf("%s@t%.1f (t=%.1f)", sm$learner[sig],
                                    sm$theta[sig], sm$t_stat_vs_zero[sig]),
                            collapse = "; ")))
cat(sprintf("signed within-rep correlations range: %+.3f to %+.3f\n",
            min(sm$cor_signed), max(sm$cor_signed)))
cat(sprintf("abs-value within-rep correlations range: %+.3f to %+.3f  <- what was reported\n",
            min(sm$cor_abs), max(sm$cor_abs)))

# Two independent routes to delta_D, and where they disagree.
#
# Route 1 (EXACT, no theory): FWL is linear and S = V_hat + R, so
#     theta_hat = FWL(V_hat) + FWL(R) = (theta + delta_D) + resid_D_coef
#   =>  delta_D = theta_bias - resid_D_coef.
# Route 2 (THEORY): bias = cbar_1 * delta_D  =>  delta_D = theta_bias/cbar_1.
#
# Agreement between the routes is a test of the pass-through model. Where
# they disagree, the linearisation has broken down and the bias is coming
# from the X-leak / higher-order terms rather than from the D-coefficient.
cat("\n== two routes to delta_D (agreement = pass-through model holds) ==\n")
cbar <- c("1.5" = 0.669, "3" = 0.861)
sm$delta_D_exact <- NA_real_; sm$delta_D_theory <- NA_real_
for (i in seq_len(nrow(sm))) {
  cb <- cbar[[as.character(sm$theta[i])]]
  sm$delta_D_exact[i]  <- sm$mean_theta_bias[i] - sm$mean_signed_resid_D[i]
  sm$delta_D_theory[i] <- sm$mean_theta_bias[i] / cb
}
cat(sprintf("%5s %-11s %12s %12s %10s\n", "theta", "learner",
            "exact", "theory", "gap"))
for (i in seq_len(nrow(sm))) {
  gap <- sm$delta_D_exact[i] - sm$delta_D_theory[i]
  cat(sprintf("%5.1f %-11s %12.4f %12.4f %10.4f%s\n", sm$theta[i],
              sm$learner[i], sm$delta_D_exact[i], sm$delta_D_theory[i],
              gap, ifelse(abs(gap) > 0.04, "  <- model breaks down", "")))
}

stopifnot(sum(sig) >= 1)                       # the signal exists
stopifnot(max(sm$cor_signed) > 0.3)            # and is not zero within-rep
cat("\nPASS: signed re-analysis recovers signal the abs-value summary destroyed.\n")
cat("NOTES / limits, both of which must survive into the write-up:\n")
cat(" 1. Shared draw stream between R and theta_hat inflates cor_signed;\n")
cat("    an independent-stream re-run is required before calling 3h positive.\n")
cat(" 2. The pass-through model holds where the index is decent but breaks\n")
cat("    down for ranger (worst index fit of any learner: R2 0.41-0.48).\n")
cat("    There the bias is carried by the residual/X-leak term rather than\n")
cat("    by delta_D, so a linear pass-through story is NOT universal --\n")
cat("    which is precisely why delta_D must be MEASURED directly rather\n")
cat("    than backed out of the bias.\n")

write.csv(sm, "R/results/stage3h_signed_summary.csv", row.names = FALSE)
cat("wrote R/results/stage3h_signed_summary.csv\n")
