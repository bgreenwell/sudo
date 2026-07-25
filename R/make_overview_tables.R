# Regenerate the results tables in manuscript/OVERVIEW.md from the committed
# stage summaries in R/results/. Idempotent; replaces the content between
# <!-- begin:tbl-* --> / <!-- end:tbl-* --> markers and touches nothing else.
# Run from repo root: Rscript R/make_overview_tables.R

read_stage <- function(n) read.csv(sprintf("R/results/stage%d_summary.csv", n))

fmt <- function(x, digits = 3, signed = FALSE) {
  ifelse(is.na(x), "",
         sprintf(if (signed) paste0("%+.", digits, "f")
                 else paste0("%.", digits, "f"), x))
}

# rows: data.frame with label + summary columns -> markdown table lines
md_table <- function(rows, note = NULL) {
  lines <- c(
    "| Config | Bias | SD | RMSE | Mean SE | Coverage |",
    "|---|---:|---:|---:|---:|---:|",
    sprintf("| %s | %s | %s | %s | %s | %s |",
            rows$label,
            fmt(rows$bias, signed = TRUE), fmt(rows$sd), fmt(rows$rmse),
            fmt(rows$mean_se), fmt(rows$coverage)))
  if (!is.null(note)) lines <- c(lines, "", note)
  lines
}

pick <- function(df, map) {
  out <- df[match(names(map), df$estimator), ]
  out$label <- unname(unlist(map))
  out
}

tables <- list()

s2 <- read_stage(2)
tables[["tbl-stage2"]] <- md_table(pick(s2, list(
  naive        = "Naive single draw",
  improper_B10 = "Improper, B=10",
  improper_B25 = "Improper, B=25",
  improper_B50 = "Improper, B=50",
  proper_B10   = "Proper MI, B=10",
  proper_B25   = "Proper MI, B=25",
  proper_B50   = "Proper MI, B=50")),
  note = "True theta = 1.5; nominal coverage 0.95.")

s3 <- read_stage(3)
tables[["tbl-stage3"]] <- md_table(pick(s3, list(
  gam_t0.5_n1000  = "gam, theta=0.5, n=1000",
  gam_t0.5_n2000  = "gam, theta=0.5, n=2000",
  gam_t3_n1000    = "gam, theta=3, n=1000",
  gam_t3_n2000    = "gam, theta=3, n=2000",
  ranger_t3_n2000 = "ranger (improper), theta=3, n=2000",
  includeD_TRUE   = "Full model with D (theta=1.5)",
  includeD_FALSE  = "Full model without D (theta=1.5)",
  refitS_FALSE    = "E[S|X] fixed across draws",
  refitS_TRUE     = "E[S|X] refit per draw")))

s3c <- read.csv("R/results/stage3c_summary.csv")
tables[["tbl-stage3c"]] <- md_table(pick(s3c, list(
  direct_glm = "Direct coefficient, glm linear in X (misspecified)",
  direct_gam = "Direct coefficient, gam (correct additive form)",
  sudo       = "SUDO (gam full model, proper draws)")),
  note = "True theta = 1.5, confounded DGP; nominal coverage 0.95.")

s3d <- read.csv("R/results/stage3d_summary.csv")
s3e <- read.csv("R/results/stage3e_summary.csv")
s3f <- read.csv("R/results/stage3f_summary.csv")

# combined view of the black-box program: one row per (learner, draw
# scheme) at each signal strength, pulled from stages 3/3d/3e/3f
bb15 <- rbind(s3d[s3d$estimator == "improper_rf_t1.5_n2000", ],
             s3d[s3d$estimator == "boot_rf_t1.5_n2000", ],
             s3e[s3e$estimator == "boot_recal_t1.5", ],
             s3f[s3f$estimator == "improper_nn_t1.5", ],
             s3f[s3f$estimator == "boot_nn_t1.5", ])
bb15$estimator <- c("improper_rf_t1.5_n2000", "boot_rf_t1.5_n2000",
                    "boot_recal_t1.5", "improper_nn_t1.5", "boot_nn_t1.5")
tables[["tbl-blackbox-t15"]] <- md_table(pick(bb15, list(
  improper_rf_t1.5_n2000 = "RF, improper draws",
  boot_rf_t1.5_n2000     = "RF, bootstrap draws",
  boot_recal_t1.5        = "RF, bootstrap + isotonic recalibration",
  improper_nn_t1.5       = "Tuned NN (decay=0.1), improper draws",
  boot_nn_t1.5           = "Tuned NN (decay=0.1), bootstrap draws")),
  note = "True theta = 1.5, n = 2000; nominal coverage 0.95.")

bb3 <- rbind(s3[s3$estimator == "ranger_t3_n2000", ],
            s3d[s3d$estimator == "boot_rf_t3_n2000", ],
            s3e[s3e$estimator == "boot_recal_t3", ],
            s3f[s3f$estimator == "improper_nn_t3", ],
            s3f[s3f$estimator == "boot_nn_t3", ])
bb3$estimator <- c("ranger_t3_n2000", "boot_rf_t3_n2000", "boot_recal_t3",
                   "improper_nn_t3", "boot_nn_t3")
tables[["tbl-blackbox-t3"]] <- md_table(pick(bb3, list(
  ranger_t3_n2000  = "RF, improper draws",
  boot_rf_t3_n2000 = "RF, bootstrap draws",
  boot_recal_t3    = "RF, bootstrap + isotonic recalibration",
  improper_nn_t3   = "Tuned NN (decay=0.1), improper draws",
  boot_nn_t3       = "Tuned NN (decay=0.1), bootstrap draws")),
  note = "True theta = 3, n = 2000; nominal coverage 0.95.")

s3g <- read.csv("R/results/stage3g_summary.csv")
fmt3 <- function(x, digits = 3, signed = FALSE) fmt(x, digits, signed)
md_calib_table <- function(df) {
  lab <- c(gam = "gam", ranger = "ranger",
          nnet_d0.01 = "nnet, decay=0.01 (untuned)",
          nnet_d0.1  = "nnet, decay=0.1 (tuned)")
  lines <- c("| Learner | theta | Slope | Intercept | R2 |",
            "|---|---:|---:|---:|---:|")
  for (th in c(1.5, 3)) {
    for (nm in names(lab)) {
      r <- df[df$learner == nm & df$theta == th, ]
      lines <- c(lines, sprintf("| %s | %g | %s | %s | %s |", lab[nm], th,
                                fmt3(r$slope, signed = TRUE),
                                fmt3(r$intercept, signed = TRUE),
                                fmt3(r$r2)))
    }
  }
  c(lines, "", paste("Slope of `lm(eta ~ V_hat)`: 1 = calibrated, <1 =",
                     "attenuated. R2 = fraction of the true index",
                     "variance explained. n = 2000, 100 reps."))
}
tables[["tbl-index-calib"]] <- md_calib_table(s3g)

s3h <- read.csv("R/results/stage3h_summary.csv")
md_resid_table <- function(df) {
  lab <- c(gam = "gam", ranger = "ranger",
          nnet_d0.01 = "nnet, decay=0.01 (untuned)",
          nnet_d0.1  = "nnet, decay=0.1 (tuned)")
  lines <- c("| Learner | theta | mean abs(resid-on-D coef) | var ratio (D=1/D=0) | mean abs(theta bias) | within-rep cor |",
            "|---|---:|---:|---:|---:|---:|")
  for (th in c(1.5, 3)) {
    for (nm in names(lab)) {
      r <- df[df$learner == nm & df$theta == th, ]
      lines <- c(lines, sprintf("| %s | %g | %s | %s | %s | %s |", lab[nm],
                                th, fmt3(r$mean_abs_resid_D),
                                fmt3(r$mean_var_ratio),
                                fmt3(r$mean_abs_theta_bias),
                                fmt3(r$within_rep_cor, signed = TRUE)))
    }
  }
  c(lines, "", paste("Single-draw (no B-loop) theta for comparability;",
                     "n = 2000, 100 reps. resid-on-D coef from",
                     "gam(R ~ D + s(X)) on the surrogate residual;",
                     "within-rep cor is Pearson cor(|coef|, |theta bias|)",
                     "across the 100 reps for that learner/theta."))
}
tables[["tbl-resid-diag"]] <- md_resid_table(s3h)

s3i <- read.csv("R/results/stage3i_summary.csv")
md_varratio_table <- function(df) {
  lab <- c(gam = "gam", ranger = "ranger",
          nnet_d0.01 = "nnet, decay=0.01 (untuned)",
          nnet_d0.1  = "nnet, decay=0.1 (tuned)")
  lines <- c("| Learner | theta | mean var ratio | t-test p (vs 1) | frac p<0.05 | mean abs(theta bias) |",
            "|---|---:|---:|---:|---:|---:|")
  for (th in c(1.5, 3)) {
    for (nm in names(lab)) {
      r <- df[df$learner == nm & df$theta == th, ]
      lines <- c(lines, sprintf("| %s | %g | %s | %s | %s | %s |", lab[nm],
                                th, fmt3(r$mean_var_ratio),
                                ifelse(r$var_ratio_ttest_p < 5e-4,
                                       sprintf("%.1e", r$var_ratio_ttest_p),
                                       fmt3(r$var_ratio_ttest_p)),
                                fmt3(r$frac_p_het_sig),
                                fmt3(r$mean_abs_theta_bias)))
    }
  }
  c(lines, "", paste("var ratio = Var(R|D=1)/Var(R|D=0); t-test p is",
                     "H0: mean ratio = 1 across the 100 reps; frac p<0.05",
                     "is the fraction of individual var.test()s significant",
                     "at 0.05 (expect ~0.05 under a true null)."))
}
tables[["tbl-varratio-diag"]] <- md_varratio_table(s3i)

# three partially-linear (PL) full model implementations: does removing
# D*X interaction fix black-box coverage? stages 3j (backfitting),
# 3k (xgboost interaction_constraints), 3l (mboost componentwise boosting)
s3j <- read.csv("R/results/stage3j_summary.csv")
s3k <- read.csv("R/results/stage3k_summary.csv")
s3l <- read.csv("R/results/stage3l_summary.csv")
pl_bias_cov <- function(theta_tag, rows) {
  lab <- c(
    "improper_plbf" = "Backfit (nnet f, untuned), improper",
    "boot_plbf"     = "Backfit (nnet f, untuned), bootstrap",
    "improper_plxgb" = "XGBoost interaction_constraints, improper",
    "boot_plxgb"     = "XGBoost interaction_constraints, bootstrap",
    "improper_plmb" = "mboost bols(D)+bbs(X), improper",
    "boot_plmb"     = "mboost bols(D)+bbs(X), bootstrap")
  df <- do.call(rbind, list(s3j, s3k, s3l))
  df <- df[df$estimator %in% paste0(names(lab), "_", theta_tag), ]
  df$key <- sub(paste0("_", theta_tag, "$"), "", df$estimator)
  df <- df[match(names(lab), df$key), ]
  df$label <- unname(lab)
  md_table(df, note = sprintf(
    paste("True theta = %s, n = 2000; nominal coverage 0.95. \"Untuned\"",
          "= same deliberately under-regularized configs that produced",
          "the worst results in experiment 6."),
    ifelse(theta_tag == "t1.5", "1.5", "3")))
}
tables[["tbl-pl-t15"]] <- pl_bias_cov("t1.5", NULL)
tables[["tbl-pl-t3"]] <- pl_bias_cov("t3", NULL)

s3j_c <- read.csv("R/results/stage3j_calib.csv")
s3k_c <- read.csv("R/results/stage3k_calib.csv")
s3l_c <- read.csv("R/results/stage3l_calib.csv")
pl_calib <- rbind(s3j_c, s3k_c, s3l_c)
md_pl_calib <- function(df) {
  lab <- c(pl_backfit = "Backfit (nnet f)",
          pl_xgboost = "XGBoost interaction_constraints",
          pl_mboost  = "mboost bols(D)+bbs(X)")
  lines <- c("| PL variant | theta | Slope | Intercept | R2 |",
            "|---|---:|---:|---:|---:|")
  for (th in c(1.5, 3)) {
    for (nm in names(lab)) {
      r <- df[df$learner == nm & df$theta == th, ]
      lines <- c(lines, sprintf("| %s | %g | %s | %s | %s |", lab[nm], th,
                                fmt3(r$slope, signed = TRUE),
                                fmt3(r$intercept, signed = TRUE),
                                fmt3(r$r2)))
    }
  }
  c(lines, "", paste("Slope of `lm(eta ~ V_hat)`: 1 = calibrated. For",
                     "reference (stage 3g): gam slope ~1/R2>=0.91; ranger",
                     "slope 0.57-0.92/R2 0.41-0.48; untuned nnet (no PL)",
                     "slope 0.65-0.72/R2 0.63-0.66."))
}
tables[["tbl-pl-calib"]] <- md_pl_calib(pl_calib)

s3j_v <- read.csv("R/results/stage3j_varratio.csv")
s3k_v <- read.csv("R/results/stage3k_varratio.csv")
s3l_v <- read.csv("R/results/stage3l_varratio.csv")
pl_vr <- rbind(s3j_v, s3k_v, s3l_v)
md_pl_vr <- function(df) {
  lab <- c(pl_backfit = "Backfit (nnet f)",
          pl_xgboost = "XGBoost interaction_constraints",
          pl_mboost  = "mboost bols(D)+bbs(X)")
  lines <- c("| PL variant | theta | mean var ratio | t-test p (vs 1) | frac p<0.05 | mean abs(theta bias) |",
            "|---|---:|---:|---:|---:|---:|")
  for (th in c(1.5, 3)) {
    for (nm in names(lab)) {
      r <- df[df$learner == nm & df$theta == th, ]
      lines <- c(lines, sprintf("| %s | %g | %s | %s | %s | %s |", lab[nm],
                                th, fmt3(r$mean_var_ratio),
                                ifelse(r$var_ratio_ttest_p < 5e-4,
                                       sprintf("%.1e", r$var_ratio_ttest_p),
                                       fmt3(r$var_ratio_ttest_p)),
                                fmt3(r$frac_p_het_sig),
                                fmt3(r$mean_abs_theta_bias)))
    }
  }
  c(lines, "", paste("Single-draw theta for comparability. For reference",
                     "(stage 3i): untuned nnet (no PL) var ratio 0.89-0.94,",
                     "p as low as 1.7e-27."))
}
tables[["tbl-pl-varratio"]] <- md_pl_vr(pl_vr)

s3m <- read.csv("R/results/stage3m_summary.csv")
tables[["tbl-pl-tuned"]] <- md_table(pick(s3m, list(
  improper_pltuned_t1.5 = "Tuned PL backfit, improper, theta=1.5",
  boot_pltuned_t1.5     = "Tuned PL backfit, bootstrap, theta=1.5",
  improper_pltuned_t3   = "Tuned PL backfit, improper, theta=3",
  boot_pltuned_t3       = "Tuned PL backfit, bootstrap, theta=3")),
  note = paste("size=4, decay=0.3, n_iter=5 (selected on MC theta bias,",
               "not a proxy); n = 2000, B = 25, 100 reps. Untuned",
               "baseline for comparison: bootstrap bias -0.023/-0.106,",
               "coverage 0.910/0.830."))

# full-pipeline bootstrap (stage 3p) vs the recentered scheme (stage 3m),
# tuned PL-backfit -- the fix that closes the theta=3 coverage gap
s3p <- read.csv("R/results/stage3p_summary.csv")
s3m_pl <- read.csv("R/results/stage3m_summary.csv")
md_pboot_table <- function() {
  bootrow <- function(th) s3m_pl[s3m_pl$estimator == paste0("boot_pltuned_t", th), ]
  prow <- function(th) s3p[s3p$theta == th, ]
  lines <- c("| Draw scheme | theta | Bias | SD | Mean SE | SD/SE | Coverage |",
            "|---|---:|---:|---:|---:|---:|---:|")
  for (th in c(1.5, 3)) {
    b <- bootrow(th); p <- prow(th)
    lines <- c(lines,
      sprintf("| Recentered within-fold bootstrap | %g | %s | %s | %s | %s | %s |",
              th, fmt3(b$bias, signed = TRUE), fmt3(b$sd), fmt3(b$mean_se),
              fmt3(b$sd / b$mean_se), fmt3(b$coverage)),
      sprintf("| Full-pipeline bootstrap | %g | %s | %s | %s | %s | %s (pct %s) |",
              th, fmt3(p$bias, signed = TRUE), fmt3(p$sd), fmt3(p$mean_se),
              fmt3(p$sd_over_mean_se), fmt3(p$coverage), fmt3(p$coverage_pct)))
  }
  c(lines, "", paste("Tuned PL-backfit imputation model; n = 2000, 100 reps.",
                     "SD/SE ~ 1 means the reported SE matches true",
                     "variability. Full-pipeline coverage shown as normal CI",
                     "(pct = percentile CI). B_outer = 100."))
}
tables[["tbl-pipeline-boot"]] <- md_pboot_table()

s3n <- read.csv("R/results/stage3n_summary.csv")
md_oracle_table <- function(df) {
  lab <- c(O0 = "O0 oracle index + oracle nuisances",
          O1 = "O1 oracle index + gam nuisances",
          O2 = "O2 gam full model (proper draws)",
          O3 = "O3 tuned nnet + bootstrap",
          O4 = "O4 tuned PL backfit + bootstrap")
  lines <- c("| Arm | theta | Bias | SD | Mean SE | SD/SE | Coverage | Known-SE cov |",
            "|---|---:|---:|---:|---:|---:|---:|---:|")
  for (th in c(1.5, 3)) {
    for (id in names(lab)) {
      r <- df[df$arm == id & df$theta == th, ]
      lines <- c(lines, sprintf("| %s | %g | %s | %s | %s | %s | %s | %s |",
                                lab[id], th, fmt3(r$bias, signed = TRUE),
                                fmt3(r$sd), fmt3(r$mean_se),
                                fmt3(r$sd_over_mean_se), fmt3(r$coverage),
                                fmt3(r$known_se_coverage)))
    }
  }
  c(lines, "", paste("n = 2000, B = 25; O0/O1 500 reps, O2-O4 200 reps.",
                     "SD/SE < 1 = conservative (SE too large); > 1 =",
                     "anti-conservative. Known-SE cov = coverage of",
                     "theta_hat +/- 1.96*mean(se); its closeness to actual",
                     "coverage means the shortfall is SE level, not noise."))
}
tables[["tbl-oracle"]] <- md_oracle_table(s3n)

s45 <- rbind(read_stage(4), read_stage(5))
tables[["tbl-ordinal"]] <- md_table(pick(s45, list(
  sudo_ordinal_clm  = "Parametric clm, n=3000",
  ordinal_dml_n1000 = "Spline clm + gam nuisances, n=1000",
  ordinal_dml_n2000 = "Spline clm + gam nuisances, n=2000")),
  note = "True theta = 1.0, J = 3; nominal coverage 0.95.")

s6 <- read_stage(6)
tables[["tbl-stage6"]] <- md_table(pick(s6, list(
  binary_logit        = "Binary, logit (wrong link)",
  binary_cloglog      = "Binary, cloglog (right link)",
  ordinal_logit       = "Ordinal, logit (wrong link)",
  ordinal_cloglog_min = "Ordinal, cloglog (right link)")))

diag <- s6[s6$estimator %in% c("vardrift_wrong", "vardrift_right"), ]
tables[["tbl-diagnostic"]] <- c(
  "| Fitted link | Variance-drift p-value |",
  "|---|---:|",
  sprintf("| logit (wrong) | %.1e |",
          diag$mean_est[diag$estimator == "vardrift_wrong"]),
  sprintf("| cloglog (right) | %.3f |",
          diag$mean_est[diag$estimator == "vardrift_right"]))

path <- "manuscript/OVERVIEW.md"
doc <- readLines(path)
for (id in names(tables)) {
  b <- grep(sprintf("^<!-- begin:%s -->$", id), doc)
  e <- grep(sprintf("^<!-- end:%s -->$", id), doc)
  stopifnot(length(b) == 1, length(e) == 1, b < e)
  doc <- c(doc[1:b], tables[[id]], doc[e:length(doc)])
}
writeLines(doc, path)
cat(sprintf("updated %d tables in %s\n", length(tables), path))
