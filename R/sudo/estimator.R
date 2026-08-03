# Full binary SUDO estimator (shared by stages 3 and 3c).
# Requires fwl.R, surrogate.R, rubin.R sourced and mgcv attached.

# cross-fitted binomial-gam full model of Y on (D, X); returns per-draw latent
# index functions: proper draws perturb each fold's coefficients by its
# Bayesian posterior N(beta_hat, Vp)
crossfit_fullmodel_gam <- function(Y, D, X, folds, include_D = TRUE) {
  d <- data.frame(Y = Y, D = D, X)
  rhs <- paste(sprintf("s(%s)", colnames(X)), collapse = " + ")
  fml <- as.formula(paste("Y ~", if (include_D) paste("D +", rhs) else rhs))
  fits <- lapply(folds, function(test) {
    m <- gam(fml, family = binomial, data = d[-test, ])
    list(test = test, lpm = predict(m, newdata = d[test, ], type = "lpmatrix"),
         beta = coef(m), V = vcov(m))
  })
  lp_hat <- numeric(length(Y))
  for (f in fits) lp_hat[f$test] <- as.numeric(f$lpm %*% f$beta)
  draw_lp <- function() {
    lp <- numeric(length(Y))
    for (f in fits)
      lp[f$test] <- as.numeric(f$lpm %*% MASS::mvrnorm(1, f$beta, f$V))
    lp
  }
  list(lp_hat = lp_hat, draw_lp = draw_lp)
}

# isotonic recalibration of out-of-fold probabilities: RF probabilities are
# compressed toward the base rate, which attenuates the logit index and
# biases theta; monotone recalibration against the observed y restores the
# probability scale before the link inverse is taken
recal_iso <- function(p, y) {
  o <- order(p)
  cal <- numeric(length(p))
  cal[o] <- isoreg(p[o], y[o])$yf
  cal
}

# cross-fitted link-scale index for a black-box (ranger/nnet) full model.
# Returns lp_hat plus fit_fold/to_lp so callers (sudo_binary's bootstrap
# branch; the index-calibration diagnostic) can reuse the exact same fit.
fit_blackbox_index <- function(d, full_model = c("ranger", "nnet"),
                               include_D = TRUE, folds,
                               recalibrate = FALSE,
                               nn_size = 8, nn_decay = 0.01) {
  full_model <- match.arg(full_model)
  feat <- data.frame(d$X, D = d$D)
  if (!include_D) feat$D <- NULL
  ydat <- factor(d$Y)
  fit_fold <- if (full_model == "ranger") {
    function(train_idx, test) {
      m <- ranger::ranger(y ~ ., data = cbind(y = ydat, feat)[train_idx, ],
                          num.trees = 200, min.node.size = 5,
                          probability = TRUE)
      predict(m, data = feat[test, , drop = FALSE])$predictions[, "1"]
    }
  } else {
    # smooth black-box: single-hidden-layer net with logistic output,
    # estimates the log-odds surface directly, avoiding the forest's
    # piecewise-constant index attenuation
    function(train_idx, test) {
      m <- nnet::nnet(x = feat[train_idx, ], y = d$Y[train_idx],
                      size = nn_size, decay = nn_decay, maxit = 500,
                      entropy = TRUE, trace = FALSE)
      as.numeric(predict(m, feat[test, , drop = FALSE]))
    }
  }
  to_lp <- function(p) {
    if (recalibrate) p <- recal_iso(p, d$Y)
    qlogis(pmin(pmax(p, 1e-5), 1 - 1e-5))
  }
  n <- nrow(feat)
  p_hat <- numeric(n)
  for (test in folds) p_hat[test] <- fit_fold(setdiff(seq_len(n), test), test)
  list(lp_hat = to_lp(p_hat), fit_fold = fit_fold, to_lp = to_lp)
}

# full binary SUDO for one dataset. For a black-box (ranger) full model,
# proper_boot = TRUE makes the draws proper by nonparametric bootstrap:
# each draw refits the forest on a within-fold resample and predicts the
# original observations, the frequentist analogue of the posterior redraw;
# recalibrate = TRUE applies isotonic recalibration to every probability
# vector before the link inverse.
# full_model may be a character name OR a function(d, folds) returning the
# same list(lp_hat, fit_fold, to_lp) contract the built-in fitters use --
# this is how oracle and deliberately-perturbed indices are injected
# (stages 3n/3o). A fitter with fit_fold = NULL has no resampling scheme,
# so bootstrap draws are switched off automatically.
# fit_l / fit_m are the partialling nuisance fitters, following the
# fitter(X, y) -> function(Xnew) contract from crossfit().
sudo_binary <- function(d, B = 25, n_folds = 5, folds = NULL, proper = TRUE,
                        full_model = c("gam", "ranger", "nnet",
                                      "pl_gam", "pl_series", "pl_backfit",
                                      "pl_xgboost", "pl_mboost", "pl_mars"),
                        include_D = TRUE,
                        refit_S_nuisance = FALSE, proper_boot = FALSE,
                        recalibrate = FALSE, nn_size = 8, nn_decay = 0.01,
                        pl_engine = "nnet", pl_use_cvrisk = TRUE,
                        pl_fixed_mstop = 200,
                        pl_n_iter = 5, mars_degree = 2, mars_nk = 41,
                        mars_penalty = 3, series_degree = 3L, link = "logit",
                        fit_l = fit_gam, fit_m = fit_gam_binomial) {
  if (is.character(full_model)) full_model <- match.arg(full_model)
  # the full model's index and the surrogate's error law must share a scale.
  # pl_gam and pl_backfit are link-aware; the rest fit on the logit scale, and pairing
  # them with another link silently rescales theta by the link ratio.
  if (link != "logit" && is.character(full_model) &&
      !full_model %in% c("pl_gam", "pl_backfit")) {
    stop("full_model '", full_model, "' fits on the logit scale; link = '",
         link, "' would mismatch the surrogate's error law")
  }
  X <- as.data.frame(d$X)
  n <- nrow(X)
  if (is.null(folds)) {
    folds <- make_folds(n, n_folds)
  } else {
    stopifnot(
      is.list(folds),
      length(folds) == n_folds,
      identical(sort(as.integer(unlist(folds))), seq_len(n))
    )
  }

  if (is.function(full_model)) {
    bb <- full_model(d, folds)
    fit_fold <- bb$fit_fold
    to_lp <- bb$to_lp
    lp_hat <- bb$lp_hat
    if (proper_boot && is.null(fit_fold)) proper_boot <- FALSE
    if (!is.null(bb$draw_lp)) {
      # the fitter carries its own proper-draw scheme (a Bayesian posterior,
      # say), so use it rather than manufacturing one by resampling
      fm <- list(lp_hat = lp_hat, draw_lp = bb$draw_lp)
    } else if (proper_boot) {
      boot_lp <- function() {
        p <- numeric(n)
        for (test in folds) {
          tr <- setdiff(seq_len(n), test)
          p[test] <- fit_fold(sample(tr, replace = TRUE), test)
        }
        to_lp(p)
      }
      LP <- sapply(seq_len(B), function(b) boot_lp())
      LP <- lp_hat + (LP - rowMeans(LP))
      i_draw <- 0
      fm <- list(lp_hat = lp_hat,
                 draw_lp = function() { i_draw <<- i_draw + 1; LP[, i_draw] })
    } else {
      fm <- list(lp_hat = lp_hat, draw_lp = function() lp_hat)
      proper <- FALSE
    }
  } else if (full_model == "gam") {
    fm <- crossfit_fullmodel_gam(d$Y, d$D, d$X, folds, include_D)
  } else {
    # every non-gam full model (ranger/nnet and the partially-linear
    # fitters) returns the same list(lp_hat, fit_fold, to_lp) shape, so the
    # bootstrap/improper draw logic below is written once and shared
    bb <- switch(full_model,
                ranger = ,
                nnet   = fit_blackbox_index(d, full_model, include_D, folds,
                                            recalibrate, nn_size, nn_decay),
                pl_gam = fit_pl_gam(d, folds, link = link),
                pl_series = fit_pl_series(d, folds, degree = series_degree),
                pl_backfit = fit_pl_backfit(d, folds, engine = pl_engine,
                                            nn_size = nn_size,
                                            nn_decay = nn_decay,
                                            n_iter = pl_n_iter,
                                            link = link),
                pl_xgboost = fit_pl_xgboost(d, folds),
                pl_mboost  = fit_pl_mboost(d, folds,
                                           use_cvrisk = pl_use_cvrisk,
                                           fixed_mstop = pl_fixed_mstop),
                pl_mars = fit_pl_mars(d, folds, degree = mars_degree,
                                      nk = mars_nk, penalty = mars_penalty))
    fit_fold <- bb$fit_fold
    to_lp <- bb$to_lp
    lp_hat <- bb$lp_hat
    if (proper_boot) {
      # bootstrap-refit predictions are location-shifted (a forest refit on
      # a resample sees ~63% unique rows and smooths harder), so raw
      # bootstrap draws bias theta. Keep the bootstrap's spread/correlation
      # but recenter at the original fit: lp_hat + (lp_boot_b - mean_b).
      boot_lp <- function() {
        p <- numeric(n)
        for (test in folds) {
          tr <- setdiff(seq_len(n), test)
          p[test] <- fit_fold(sample(tr, replace = TRUE), test)
        }
        to_lp(p)
      }
      LP <- sapply(seq_len(B), function(b) boot_lp())
      LP <- lp_hat + (LP - rowMeans(LP))
      i_draw <- 0
      draw_lp <- function() {
        i_draw <<- i_draw + 1
        LP[, i_draw]
      }
      fm <- list(lp_hat = lp_hat, draw_lp = draw_lp)
    } else {
      fm <- list(lp_hat = lp_hat, draw_lp = function() lp_hat)  # improper
      proper <- FALSE
    }
  }

  D_res <- d$D - crossfit(X, d$D, fit_m, folds)

  # outcome nuisance E[S|X] on an initial draw, reused across draws unless
  # refit_S_nuisance
  S_init <- complete_surrogate(d$Y + 1L, fm$lp_hat, c(-Inf, 0, Inf), link)
  S_hat <- crossfit(X, S_init, fit_l, folds)

  draws <- sapply(seq_len(B), function(b) {
    lp_b <- if (proper) fm$draw_lp() else fm$lp_hat
    S_b <- complete_surrogate(d$Y + 1L, lp_b, c(-Inf, 0, Inf), link)
    S_hat_b <- if (refit_S_nuisance) crossfit(X, S_b, fit_l, folds) else S_hat
    f <- fwl_theta(S_b - S_hat_b, D_res)
    c(theta = f$theta, var = f$var)
  })
  p <- pool_rubin(draws["theta", ], draws["var", ], n_obs = n)
  # W / B_between / df are carried out so MC drivers can decompose where a
  # coverage shortfall comes from (SE level vs SE variability)
  list(theta = p$theta, se = p$se, ci_lo = p$ci_lo, ci_hi = p$ci_hi,
       W = p$W, B_between = p$B_between, df = p$df,
       max_abs_lp = max(abs(fm$lp_hat)))
}

# Full-pipeline nonparametric bootstrap of the whole SUDO estimator.
#
# The recentered within-fold bootstrap (proper_boot=TRUE) perturbs only the
# imputation-model index and holds the cross-fitting folds and the
# partialling nuisances fixed; the oracle control (stage 3n) showed it
# under-propagates the full model's own sampling variance into the SE. This
# instead resamples the ENTIRE dataset and reruns the COMPLETE pipeline per
# replicate, so the spread across replicates captures full-model estimation,
# partialling nuisances, and surrogate noise. With fold_mode = "redraw" it
# also propagates fold assignment; fold_mode = "fixed" conditions on one
# regular partition. The SE is the SD of theta across replicates; the point
# estimate uses the fitted index (a point estimate does not need proper
# parameter draws). `...` is forwarded to sudo_binary, so any full_model /
# pl_* configuration works.
#
# Caveat: bootstrap ties duplicate rows, which can place copies of one
# observation in both a cross-fit training and test fold (mild optimism) --
# the standard caveat for bootstrapping cross-fitted estimators.
sudo_pipeline_boot <- function(d, B_outer = 100, inner_B = 15,
                               level = 0.95,
                               fold_mode = c("redraw", "fixed"),
                               n_folds = 5, estimator = sudo_binary,
                               resample_indices = NULL, fold_builder = NULL,
                               base_folds = NULL, bootstrap_indices = NULL,
                               bootstrap_folds = NULL,
                               ...) {
  fold_mode <- match.arg(fold_mode)
  n <- length(d$Y)
  stopifnot(B_outer >= 2L, inner_B >= 2L, n_folds >= 2L)
  Xd <- as.data.frame(d$X)
  fixed_folds <- if (!is.null(base_folds)) base_folds else
    if (fold_mode == "fixed") make_folds(n, n_folds) else NULL
  point <- function(dd, folds = NULL) {
    if (is.null(folds) && !is.null(fold_builder)) {
      folds <- fold_builder(dd, n_folds)
    }
    args <- c(list(d = dd, B = inner_B, n_folds = n_folds, folds = folds),
              list(...))
    if (identical(estimator, sudo_binary)) args$proper_boot <- FALSE
    do.call(estimator, args)
  }
  fit0 <- point(d, fixed_folds)
  extract_targets <- function(fit) {
    value <- if (is.null(fit$targets)) c(theta = fit$theta) else fit$targets
    stopifnot(is.numeric(value), length(value) >= 1L,
              !is.null(names(value)), !anyDuplicated(names(value)))
    value
  }
  extract_internal_se <- function(fit, target_names) {
    value <- if (is.null(fit$target_internal_se)) c(theta = fit$se) else
      fit$target_internal_se
    out <- rep(NA_real_, length(target_names))
    names(out) <- target_names
    out[intersect(names(value), target_names)] <-
      value[intersect(names(value), target_names)]
    out
  }
  target0 <- extract_targets(fit0)
  boot_list <- lapply(seq_len(B_outer), function(b) {
    idx <- if (!is.null(bootstrap_indices)) bootstrap_indices[[b]] else
      if (is.null(resample_indices)) sample.int(n, replace = TRUE) else
        resample_indices(d, b)
    stopifnot(length(idx) >= 2L, all(idx >= 1L), all(idx <= n))
    boot_fold <- if (!is.null(bootstrap_folds)) bootstrap_folds[[b]] else
      fixed_folds
    fit <- point(
      list(X = Xd[idx, , drop = FALSE], D = d$D[idx], Y = d$Y[idx]),
      boot_fold)
    targets <- extract_targets(fit)
    if (!identical(names(targets), names(target0))) {
      stop("bootstrap target names changed across pipeline fits")
    }
    list(
      targets = targets,
      internal_se = extract_internal_se(fit, names(target0)),
      thresholds_ordered = if (is.null(fit$thresholds_ordered)) 1 else
        fit$thresholds_ordered,
      min_threshold_gap = if (is.null(fit$min_threshold_gap)) Inf else
        fit$min_threshold_gap
    )
  })
  boot_targets <- do.call(cbind, lapply(boot_list, `[[`, "targets"))
  boot_internal <- do.call(cbind, lapply(boot_list, `[[`, "internal_se"))
  target_se <- apply(boot_targets, 1L, sd)
  target_center <- rowMeans(boot_targets)
  target_internal0 <- extract_internal_se(fit0, names(target0))
  z <- qnorm(1 - (1 - level) / 2)
  a <- (1 - level) / 2
  target_results <- data.frame(
    target = names(target0), estimate = as.numeric(target0),
    se = as.numeric(target_se),
    ci_lo = as.numeric(target0 - z * target_se),
    ci_hi = as.numeric(target0 + z * target_se),
    bootstrap_center = as.numeric(target_center),
    internal_se = as.numeric(target_internal0), row.names = NULL
  )
  primary <- names(target0)[1]
  th0 <- target0[[primary]]
  thb <- boot_targets[primary, ]
  se <- target_se[[primary]]
  t_boot <- (thb - th0) / boot_internal[primary, ]
  t_quantile <- quantile(t_boot[is.finite(t_boot)], c(a, 1 - a))
  out <- list(theta = th0, se = se, ci_lo = th0 - z * se, ci_hi = th0 + z * se,
       ci_lo_pct = unname(quantile(thb, a)),
       ci_hi_pct = unname(quantile(thb, 1 - a)),
       ci_lo_stud = th0 - unname(t_quantile[2]) * fit0$se,
       ci_hi_stud = th0 - unname(t_quantile[1]) * fit0$se,
       internal_se = target_internal0[[primary]], bootstrap_center = mean(thb),
       targets = target0, target_se = target_se,
       target_results = target_results, bootstrap_targets = boot_targets,
       B_outer = B_outer, fold_mode = fold_mode)
  if (!is.null(fit0$direct_theta)) out$direct_theta <- fit0$direct_theta
  out$all_boot_thresholds_ordered <-
    as.numeric(all(vapply(boot_list, `[[`, numeric(1),
                          "thresholds_ordered") == 1))
  out$min_boot_threshold_gap <- min(vapply(
    boot_list, `[[`, numeric(1), "min_threshold_gap"
  ))
  diagnostics <- setdiff(names(fit0), c(names(out), "ci_lo", "ci_hi",
                                        "theta", "se"))
  for (name in diagnostics) {
    value <- fit0[[name]]
    if (is.numeric(value) && length(value) == 1L) out[[name]] <- value
  }
  out
}
