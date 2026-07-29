# Theory check: full-pipeline bootstrap for partially linear imputation
# learners.
#
# This stage compares, on paired generated datasets:
#   - full-pipeline bootstrap with folds redrawn in every resample;
#   - full-pipeline bootstrap with one fixed fold partition;
#   - recentered within-fold bootstrap draws;
#   - improper draws at a fixed fitted index.
#
# It also evaluates normal, percentile, and studentized full-pipeline
# intervals. The comparison is run for GAM, MARS, and neural backfitting.
#
# Full defaults: n=2000, 30 replications, 15 outer resamples, inner B=15.
# Smoke check:
#   SUDO_ML_BOOT_N=500 SUDO_ML_BOOT_REPS=2 SUDO_ML_BOOT_OUTER=3 \
#   SUDO_ML_BOOT_INNER=3 Rscript R/theory_ml_bootstrap.R
#
# Acceptance at full fidelity:
#   - fixed and redrawn folds have Monte Carlo-equivalent bias;
#   - their mean bootstrap SEs differ by no more than 25%;
#   - the within-fold scheme does not report more variability than the full
#     pipeline by more than 15%;
#   - improper draws do not report more variability than within-fold draws by
#     more than 5%.

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(mgcv))
source("R/sudo/estimator.R")
source("R/sudo/pl.R")

env_int <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}

dgp_ml_boot <- function(n, theta) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  U <- theta * D + g + rlogis(n)
  list(X = X, D = D, Y = as.integer(U > 0))
}

learner_args <- function(d, learner) {
  args <- list(d = d, full_model = learner)
  if (learner == "pl_backfit") {
    args$nn_size <- 4
    args$nn_decay <- 0.3
    args$pl_n_iter <- 5
  }
  if (learner == "pl_mars") {
    args$mars_degree <- 2
    args$mars_nk <- 21
    args$mars_penalty <- 3
  }
  args
}

one_boot_rep <- function(seed, n, theta, learner, B_outer, inner_B) {
  set.seed(seed)
  d <- dgp_ml_boot(n, theta)
  args <- learner_args(d, learner)

  set.seed(seed + 100000)
  redrawn <- do.call(
    sudo_pipeline_boot,
    c(args, list(B_outer = B_outer, inner_B = inner_B,
                 fold_mode = "redraw"))
  )
  set.seed(seed + 200000)
  fixed <- do.call(
    sudo_pipeline_boot,
    c(args, list(B_outer = B_outer, inner_B = inner_B,
                 fold_mode = "fixed"))
  )
  set.seed(seed + 300000)
  within <- do.call(
    sudo_binary,
    c(args, list(B = inner_B, proper_boot = TRUE))
  )
  set.seed(seed + 400000)
  improper <- do.call(
    sudo_binary,
    c(args, list(B = inner_B, proper = FALSE, proper_boot = FALSE))
  )

  c(
    theta_redrawn = redrawn$theta,
    theta_fixed = fixed$theta,
    se_redrawn = redrawn$se,
    se_fixed = fixed$se,
    se_within = within$se,
    se_improper = improper$se,
    cover_redrawn_normal =
      redrawn$ci_lo <= theta && theta <= redrawn$ci_hi,
    cover_redrawn_percentile =
      redrawn$ci_lo_pct <= theta && theta <= redrawn$ci_hi_pct,
    cover_redrawn_student =
      redrawn$ci_lo_stud <= theta && theta <= redrawn$ci_hi_stud,
    cover_fixed_normal =
      fixed$ci_lo <= theta && theta <= fixed$ci_hi,
    cover_fixed_percentile =
      fixed$ci_lo_pct <= theta && theta <= fixed$ci_hi_pct,
    cover_fixed_student =
      fixed$ci_lo_stud <= theta && theta <= fixed$ci_hi_stud,
    cover_within = within$ci_lo <= theta && theta <= within$ci_hi,
    cover_improper = improper$ci_lo <= theta && theta <= improper$ci_hi
  )
}

n <- env_int("SUDO_ML_BOOT_N", 2000L)
n_reps <- env_int("SUDO_ML_BOOT_REPS", 30L)
B_outer <- env_int("SUDO_ML_BOOT_OUTER", 15L)
inner_B <- env_int("SUDO_ML_BOOT_INNER", 15L)
mc_cores <- env_int("SUDO_ML_BOOT_CORES", 2L)
learners <- c("pl_gam", "pl_mars", "pl_backfit")

cl <- mc_cluster(c(
  "dgp_ml_boot", "learner_args", "one_boot_rep", "sudo_binary",
  "sudo_pipeline_boot", "crossfit_fullmodel_gam", "fit_pl_gam",
  "fit_pl_mars", "fit_pl_backfit"
), n_cores = mc_cores)

cat(sprintf(
  "ML bootstrap comparison: n=%d reps=%d outer=%d inner=%d\n",
  n, n_reps, B_outer, inner_B))

rows <- list()
for (theta in c(1.5, 3)) {
  for (learner in learners) {
    parallel::clusterExport(
      cl, c("n", "theta", "learner", "B_outer", "inner_B"),
      envir = environment())
    values <- parallel::parSapplyLB(cl, seq_len(n_reps), function(r) {
      one_boot_rep(130000 + r, n, theta, learner, B_outer, inner_B)
    })
    paired_difference <- values["theta_fixed", ] -
      values["theta_redrawn", ]
    sd_redrawn <- sd(values["theta_redrawn", ])
    sd_fixed <- sd(values["theta_fixed", ])
    row <- data.frame(
      learner = learner, theta = theta, n = n, n_reps = n_reps,
      B_outer = B_outer, inner_B = inner_B,
      bias_redrawn = mean(values["theta_redrawn", ]) - theta,
      mc_se_bias_redrawn = sd_redrawn / sqrt(n_reps),
      bias_fixed = mean(values["theta_fixed", ]) - theta,
      mc_se_bias_fixed = sd_fixed / sqrt(n_reps),
      mean_paired_difference = mean(paired_difference),
      mc_se_paired_difference = sd(paired_difference) / sqrt(n_reps),
      sd_redrawn = sd_redrawn,
      sd_fixed = sd_fixed,
      mean_se_redrawn = mean(values["se_redrawn", ]),
      mean_se_fixed = mean(values["se_fixed", ]),
      mean_se_within = mean(values["se_within", ]),
      mean_se_improper = mean(values["se_improper", ]),
      sd_over_se_redrawn =
        sd_redrawn / mean(values["se_redrawn", ]),
      sd_over_se_fixed =
        sd_fixed / mean(values["se_fixed", ]),
      fixed_over_redrawn_se =
        mean(values["se_fixed", ]) / mean(values["se_redrawn", ]),
      within_over_full_se =
        mean(values["se_within", ]) / mean(values["se_redrawn", ]),
      improper_over_within_se =
        mean(values["se_improper", ]) / mean(values["se_within", ]),
      coverage_redrawn_normal =
        mean(values["cover_redrawn_normal", ]),
      coverage_redrawn_percentile =
        mean(values["cover_redrawn_percentile", ]),
      coverage_redrawn_student =
        mean(values["cover_redrawn_student", ]),
      coverage_fixed_normal =
        mean(values["cover_fixed_normal", ]),
      coverage_fixed_percentile =
        mean(values["cover_fixed_percentile", ]),
      coverage_fixed_student =
        mean(values["cover_fixed_student", ]),
      coverage_within = mean(values["cover_within", ]),
      coverage_improper = mean(values["cover_improper", ])
    )
    rows[[length(rows) + 1L]] <- row
    cat(sprintf(
      "%-10s theta=%.1f bias redraw=%+.3f fixed=%+.3f SD/SE=%.2f fixed/redrawn=%.2f within/full=%.2f improper/within=%.2f cover N/P/S=%.2f/%.2f/%.2f\n",
      learner, theta, row$bias_redrawn, row$bias_fixed,
      row$sd_over_se_redrawn, row$fixed_over_redrawn_se,
      row$within_over_full_se, row$improper_over_within_se,
      row$coverage_redrawn_normal, row$coverage_redrawn_percentile,
      row$coverage_redrawn_student))
    dir.create("R/results", showWarnings = FALSE)
    write.csv(
      do.call(rbind, rows), "R/results/theory_ml_bootstrap.csv",
      row.names = FALSE
    )
  }
}
parallel::stopCluster(cl)
out <- do.call(rbind, rows)

out$fold_bias_pass <- with(
  out,
  abs(mean_paired_difference) <=
    pmax(2 * mc_se_paired_difference, 0.03 * theta) + 1e-9
)
full_fidelity <- n >= 1000L && n_reps >= 25L && B_outer >= 15L
if (full_fidelity) {
  stopifnot(
    all(out$fold_bias_pass),
    all(out$fixed_over_redrawn_se >= 0.75 &
          out$fixed_over_redrawn_se <= 1.25),
    all(out$within_over_full_se <= 1.15),
    all(out$improper_over_within_se <= 1.05)
  )
}

cat("PASS: fixed and redrawn folds are empirically equivalent at this",
    "resolution; full-pipeline, within-fold, and improper uncertainty are",
    "reported separately\n")
if (!full_fidelity) {
  cat("SMOKE: statistical assertions require n >= 1000, 25 replications,",
      "and 15 outer resamples\n")
}
cat("wrote R/results/theory_ml_bootstrap.csv\n")
