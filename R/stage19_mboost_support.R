# Experimental support for stage 19. Keep these functions outside R/sudo/
# until the full target-level and coverage gates pass.

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/estimator.R")
source("R/sudo/discrete.R")

stage19_require_packages <- function() {
  required <- c("mboost", "ordinal")
  missing <- required[!vapply(
    required, requireNamespace, logical(1), quietly = TRUE
  )]
  if (length(missing)) {
    stop("stage 19 requires: ", paste(missing, collapse = ", "))
  }
  invisible(TRUE)
}

stage19_env_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}

stage19_dgp <- function(n, p = 50L, theta = 1) {
  stopifnot(n >= 100L, p >= 4L, theta > 0)
  X <- matrix(rnorm(n * p), n, p)
  colnames(X) <- paste0("X", seq_len(p))
  g0 <- 0.65 * X[, 1] + 0.55 * sin(X[, 2]) +
    0.45 * (X[, 3]^2 - 1) - 0.40 * X[, 4]
  m0 <- plogis(0.45 * X[, 1] - 0.35 * X[, 4])
  D <- rbinom(n, 1, m0)
  cuts <- c(-1.5, -0.3, 0.8, 1.8)
  eta <- theta * D + g0
  latent <- eta + rlogis(n)
  Y <- 1L + findInterval(latent, cuts)
  list(
    X = X, D = D, Y = Y, J = 5L, theta = theta, cuts = cuts,
    g0 = g0, m0 = m0, eta = eta, latent = latent
  )
}

stage19_binary_dgp <- function(n, p = 50L, theta = 1) {
  d <- stage19_dgp(n, p, theta)
  probability <- -expm1(-exp(pmin(d$eta, 30)))
  d$Y <- rbinom(n, 1, probability)
  d$probability <- probability
  d$J <- NULL
  d$cuts <- NULL
  d$latent <- NULL
  d
}

stage19_make_folds <- function(n, k, seed) {
  set.seed(seed)
  split(sample.int(n), rep_len(seq_len(k), n))
}

stage19_logistic_primitive <- function(probability) {
  out <- numeric(length(probability))
  interior <- probability > 0 & probability < 1
  p <- probability[interior]
  out[interior] <- p * log(p) + (1 - p) * log1p(-p)
  out
}

stage19_logistic_jump_mean <- function(lower, upper, clamp = 1e-12) {
  lower <- as.numeric(lower)
  upper <- as.numeric(upper)
  stopifnot(
    length(lower) == length(upper), all(is.finite(lower)),
    all(is.finite(upper)), all(lower >= 0), all(upper <= 1),
    all(lower <= upper), clamp > 0, clamp < 0.5
  )
  width <- upper - lower
  midpoint <- (lower + upper) / 2
  out <- qlogis(pmin(pmax(midpoint, clamp), 1 - clamp))
  regular <- width > sqrt(.Machine$double.eps)
  if (any(regular)) {
    lo <- pmax(lower[regular], clamp)
    hi <- pmin(upper[regular], 1 - clamp)
    integral <- numeric(length(lo))
    use <- hi > lo
    integral[use] <- stage19_logistic_primitive(hi[use]) -
      stage19_logistic_primitive(lo[use])
    lower_mass <- pmax(pmin(upper[regular], clamp) - lower[regular], 0)
    upper_mass <- pmax(
      upper[regular] - pmax(lower[regular], 1 - clamp), 0
    )
    out[regular] <- (
      lower_mass * qlogis(clamp) + integral +
        upper_mass * qlogis(1 - clamp)
    ) / width[regular]
  }
  if (!all(is.finite(out))) stop("non-finite logistic jump mean")
  out
}

stage19_formula <- function(x_names, include_treatment = TRUE,
                            response = "Y", knots = 10L, df = 4) {
  terms <- sprintf("bbs(%s, knots=%d, df=%s)", x_names, knots, df)
  if (include_treatment) terms <- c("bols(D)", terms)
  formula <- as.formula(paste(response, "~", paste(terms, collapse = " + ")))
  environment(formula) <- asNamespace("mboost")
  formula
}

stage19_config <- function(
    method = c("pure", "backfit"), mstop = 2000L, nu = 0.1,
    nuisance_mstop = 2000L, nuisance_nu = 0.1, knots = 10L, df = 4,
    backfit_tolerance = 1e-4, backfit_max_iterations = 10L, J = 5L) {
  method <- match.arg(method)
  list(
    method = method, mstop = as.integer(mstop), nu = as.numeric(nu),
    nuisance_mstop = as.integer(nuisance_mstop),
    nuisance_nu = as.numeric(nuisance_nu), knots = as.integer(knots),
    df = as.numeric(df), backfit_tolerance = backfit_tolerance,
    backfit_max_iterations = as.integer(backfit_max_iterations),
    J = as.integer(J)
  )
}

stage19_validate_ordinal_data <- function(d, J = 5L) {
  stopifnot(J == 5L, all(d$Y %in% seq_len(J)))
  if (length(unique(d$Y)) != J) stop("an ordinal category is absent")
  invisible(TRUE)
}

stage19_extract_components <- function(model, newdata = NULL) {
  used <- sort(unique(mboost::selected(model)))
  n <- if (is.null(newdata)) length(stats::fitted(model)) else nrow(newdata)
  if (!length(used)) return(rep(0, n))
  contribution <- suppressWarnings(if (is.null(newdata)) {
    predict(model, which = used, type = "link")
  } else {
    predict(model, newdata = newdata, which = used, type = "link")
  })
  if (is.list(contribution)) {
    contribution <- do.call(cbind, lapply(contribution, as.numeric))
  }
  contribution <- as.matrix(contribution)
  if (nrow(contribution) != n && ncol(contribution) == n) {
    contribution <- t(contribution)
  }
  stopifnot(nrow(contribution) == n)
  rowSums(contribution)
}

stage19_pure_spec <- function(model, test_data, y_test) {
  probability <- as.matrix(suppressWarnings(predict(
    model, newdata = test_data, type = "response"
  )))
  if (ncol(probability) != 5L) {
    stop("PropOdds did not return five category probabilities")
  }
  if (any(!is.finite(probability)) || any(probability < -1e-10) ||
      max(abs(rowSums(probability) - 1)) > 1e-6) {
    stop("invalid PropOdds category probabilities")
  }
  probability <- pmax(probability, 0)
  probability <- probability / rowSums(probability)
  bounds <- categorical_cdf_bounds(y_test, probability)
  index <- as.numeric(suppressWarnings(predict(
    model, newdata = test_data, type = "link"
  )))
  treated <- test_data
  control <- test_data
  treated$D <- 1
  control$D <- 0
  contrast <- as.numeric(suppressWarnings(
    predict(model, newdata = treated, type = "link") -
      predict(model, newdata = control, type = "link")
  ))
  cuts <- as.numeric(mboost::nuisance(model))
  list(
    index = index, lower = bounds$lower, upper = bounds$upper,
    direct_theta = mean(contrast), contrast_spread = sd(contrast),
    cuts = cuts, min_threshold_gap = min(diff(cuts)),
    probabilities = probability
  )
}

stage19_fit_pure_path <- function(d, folds, mstop_grid, nu = 0.1,
                                  knots = 10L, df = 4) {
  stage19_validate_ordinal_data(d)
  mstop_grid <- sort(unique(as.integer(mstop_grid)))
  stopifnot(all(mstop_grid > 0))
  X <- as.data.frame(d$X)
  dat <- data.frame(
    Y = ordered(d$Y, levels = seq_len(5L)), D = d$D, X,
    check.names = FALSE
  )
  formula <- stage19_formula(names(X), TRUE, "Y", knots, df)
  n <- nrow(dat)
  output <- lapply(mstop_grid, function(step) list(
    index = numeric(n), lower = numeric(n), upper = numeric(n),
    direct = numeric(length(folds)), contrast_spread = numeric(length(folds)),
    min_gap = numeric(length(folds)), max_probability_error = 0
  ))
  names(output) <- as.character(mstop_grid)
  for (fold in seq_along(folds)) {
    test <- folds[[fold]]
    train <- setdiff(seq_len(n), test)
    if (length(unique(d$Y[train])) != 5L) {
      stop("an ordinal category is absent from a training fold")
    }
    maximum <- suppressWarnings(mboost::gamboost(
      formula, data = dat[train, , drop = FALSE], family = mboost::PropOdds(),
      baselearner = mboost::bbs,
      control = mboost::boost_control(
        mstop = max(mstop_grid), nu = nu, trace = FALSE
      )
    ))
    for (step in mstop_grid) {
      model <- maximum[step]
      spec <- stage19_pure_spec(
        model, dat[test, , drop = FALSE], d$Y[test]
      )
      key <- as.character(step)
      output[[key]]$index[test] <- spec$index
      output[[key]]$lower[test] <- spec$lower
      output[[key]]$upper[test] <- spec$upper
      output[[key]]$direct[fold] <- spec$direct_theta
      output[[key]]$contrast_spread[fold] <- spec$contrast_spread
      output[[key]]$min_gap[fold] <- spec$min_threshold_gap
      output[[key]]$max_probability_error <- max(
        output[[key]]$max_probability_error,
        max(abs(rowSums(spec$probabilities) - 1))
      )
    }
  }
  lapply(output, function(value) list(
    index = value$index, lower = value$lower, upper = value$upper,
    reference = "logistic", direct_theta = mean(value$direct),
    diagnostics = list(
      direct_theta_sd = sd(value$direct),
      thresholds_ordered = as.numeric(all(value$min_gap > 0)),
      min_threshold_gap = min(value$min_gap),
      max_contrast_spread = max(value$contrast_spread),
      max_probability_error = value$max_probability_error,
      backfit_iterations = 0,
      backfit_converged = 1
    )
  ))
}

stage19_fit_backfit_fold <- function(d, train, test, config) {
  X <- as.data.frame(d$X)
  train_data <- data.frame(
    Y = ordered(d$Y[train], levels = seq_len(5L)),
    D = d$D[train], X[train, , drop = FALSE], check.names = FALSE
  )
  test_data <- data.frame(
    Y = ordered(d$Y[test], levels = seq_len(5L)),
    D = d$D[test], X[test, , drop = FALSE], check.names = FALSE
  )
  if (length(unique(train_data$Y)) != 5L) {
    stop("an ordinal category is absent from a training fold")
  }
  initial <- ordinal::clm(Y ~ D, data = train_data, link = "logit")
  alpha <- unname(initial$beta["D"])
  formula <- stage19_formula(
    names(X), FALSE, "Y", config$knots, config$df
  )
  converged <- FALSE
  iterations <- 0L
  for (iteration in seq_len(config$backfit_max_iterations)) {
    model <- suppressWarnings(mboost::gamboost(
      formula, data = train_data, family = mboost::PropOdds(),
      offset = alpha * train_data$D, baselearner = mboost::bbs,
      control = mboost::boost_control(
        mstop = config$mstop, nu = config$nu, trace = FALSE
      )
    ))
    f_train <- stage19_extract_components(model)
    f_test <- stage19_extract_components(model, test_data)
    refit_data <- train_data
    refit_data$f_offset <- f_train
    refit <- ordinal::clm(
      Y ~ D + offset(f_offset), data = refit_data, link = "logit",
      control = ordinal::clm.control(maxIter = 300L, gradTol = 1e-6)
    )
    if (!identical(refit$convergence$code, 0L) ||
        any(!is.finite(refit$coefficients))) {
      stop("backfit ordinal treatment update did not converge")
    }
    next_alpha <- unname(refit$beta["D"])
    iterations <- iteration
    if (abs(next_alpha - alpha) < config$backfit_tolerance) {
      alpha <- next_alpha
      converged <- TRUE
      break
    }
    alpha <- next_alpha
  }
  if (!converged) stop("ordinal mboost backfit did not converge")
  cuts <- as.numeric(refit$alpha)
  index <- f_test + alpha * test_data$D
  lower_cut <- c(-Inf, cuts)[d$Y[test]]
  upper_cut <- c(cuts, Inf)[d$Y[test]]
  lower <- plogis(lower_cut - index)
  upper <- plogis(upper_cut - index)
  list(
    index = index, lower = lower, upper = upper,
    direct_theta = alpha, min_threshold_gap = min(diff(cuts)),
    contrast_spread = 0, iterations = iterations,
    condition_number = refit$cond.H,
    max_gradient = refit$maxGradient
  )
}

stage19_fit_backfit <- function(d, folds, config) {
  stage19_validate_ordinal_data(d)
  n <- length(d$Y)
  index <- lower <- upper <- numeric(n)
  diagnostics <- vector("list", length(folds))
  for (fold in seq_along(folds)) {
    test <- folds[[fold]]
    train <- setdiff(seq_len(n), test)
    fit <- stage19_fit_backfit_fold(d, train, test, config)
    index[test] <- fit$index
    lower[test] <- fit$lower
    upper[test] <- fit$upper
    diagnostics[[fold]] <- unlist(fit[c(
      "direct_theta", "min_threshold_gap", "contrast_spread",
      "iterations", "condition_number", "max_gradient"
    )])
  }
  diagnostics <- do.call(rbind, diagnostics)
  list(
    index = index, lower = lower, upper = upper, reference = "logistic",
    direct_theta = mean(diagnostics[, "direct_theta"]),
    diagnostics = list(
      direct_theta_sd = sd(diagnostics[, "direct_theta"]),
      thresholds_ordered = as.numeric(
        all(diagnostics[, "min_threshold_gap"] > 0)
      ),
      min_threshold_gap = min(diagnostics[, "min_threshold_gap"]),
      max_contrast_spread = max(diagnostics[, "contrast_spread"]),
      max_probability_error = 0,
      backfit_iterations = max(diagnostics[, "iterations"]),
      backfit_converged = 1,
      max_condition_number = max(diagnostics[, "condition_number"]),
      max_gradient = max(diagnostics[, "max_gradient"])
    )
  )
}

stage19_fit_full <- function(d, folds, config) {
  if (config$method == "pure") {
    stage19_fit_pure_path(
      d, folds, config$mstop, config$nu, config$knots, config$df
    )[[1L]]
  } else {
    stage19_fit_backfit(d, folds, config)
  }
}

stage19_fit_nuisance <- function(X, y, outcome, mstop, nu, knots, df) {
  X <- as.data.frame(X)
  if (outcome == "treatment") {
    dat <- data.frame(y = factor(y, levels = c(0, 1)), X, check.names = FALSE)
    family <- mboost::Binomial(type = "glm", link = "logit")
  } else {
    dat <- data.frame(y = as.numeric(y), X, check.names = FALSE)
    family <- mboost::Gaussian()
  }
  formula <- stage19_formula(names(X), FALSE, "y", knots, df)
  model <- suppressWarnings(mboost::gamboost(
    formula, data = dat, family = family, baselearner = mboost::bbs,
    control = mboost::boost_control(mstop = mstop, nu = nu, trace = FALSE)
  ))
  function(Xnew) {
    value <- suppressWarnings(predict(
      model, newdata = as.data.frame(Xnew),
      type = if (outcome == "treatment") "response" else "link"
    ))
    as.numeric(value)
  }
}

stage19_nuisance_fitter <- function(outcome, config) {
  force(outcome)
  force(config)
  function(X, y) stage19_fit_nuisance(
    X, y, outcome, config$nuisance_mstop, config$nuisance_nu,
    config$knots, config$df
  )
}

stage19_codings <- function() list(
  equal = c(0, 2, 4, 6, 8),
  middle = c(0, 0.5, 4, 7.5, 8)
)

stage19_category_probabilities <- function(eta, cuts) {
  cumulative <- sapply(cuts, function(cut) plogis(cut - eta))
  cbind(
    cumulative[, 1],
    cumulative[, 2] - cumulative[, 1],
    cumulative[, 3] - cumulative[, 2],
    cumulative[, 4] - cumulative[, 3],
    1 - cumulative[, 4]
  )
}

stage19_oracle_coding <- function(d, scores) {
  p0 <- stage19_category_probabilities(d$g0, d$cuts)
  p1 <- stage19_category_probabilities(d$theta + d$g0, d$cuts)
  mean0 <- as.numeric(p0 %*% scores)
  mean1 <- as.numeric(p1 %*% scores)
  ell <- (1 - d$m0) * mean0 + d$m0 * mean1
  value <- scores[d$Y]
  fwl_theta(value - ell, d$D - d$m0)$theta
}

stage19_theta_from_full <- function(d, folds, full, config,
                                    include_raw = TRUE) {
  X <- as.data.frame(d$X)
  surrogate <- full$index +
    stage19_logistic_jump_mean(full$lower, full$upper)
  treatment_fitter <- stage19_nuisance_fitter("treatment", config)
  outcome_fitter <- stage19_nuisance_fitter("outcome", config)
  D_res <- d$D - crossfit(X, d$D, treatment_fitter, folds)
  nuisance <- crossfit(X, surrogate, outcome_fitter, folds)
  sudo <- fwl_theta(surrogate - nuisance, D_res)
  raw <- if (include_raw) lapply(stage19_codings(), function(scores) {
    value <- scores[d$Y]
    ell <- crossfit(X, value, outcome_fitter, folds)
    fwl_theta(value - ell, D_res)
  }) else NULL
  list(sudo = sudo, raw = raw, surrogate = surrogate, D_res = D_res)
}

stage19_point <- function(d, config, B = 2L, n_folds = 5L, folds = NULL,
                          level = 0.95, include_raw = TRUE) {
  stage19_require_packages()
  n <- length(d$Y)
  if (is.null(folds)) folds <- make_folds(n, n_folds)
  stopifnot(length(folds) == n_folds)
  full <- stage19_fit_full(d, folds, config)
  estimate <- stage19_theta_from_full(d, folds, full, config, include_raw)
  raw <- estimate$raw
  targets <- c(
    rao_blackwell = estimate$sudo$theta,
    direct_ordinal = full$direct_theta
  )
  internal <- c(
    rao_blackwell = estimate$sudo$se,
    direct_ordinal = NA_real_
  )
  if (include_raw) {
    targets <- c(
      targets, dml_equal = raw$equal$theta, dml_middle = raw$middle$theta
    )
    internal <- c(
      internal, dml_equal = raw$equal$se, dml_middle = raw$middle$se
    )
  }
  z <- qnorm(1 - (1 - level) / 2)
  out <- list(
    theta = estimate$sudo$theta, se = estimate$sudo$se,
    ci_lo = estimate$sudo$theta - z * estimate$sudo$se,
    ci_hi = estimate$sudo$theta + z * estimate$sudo$se,
    targets = targets, target_internal_se = internal,
    direct_theta = full$direct_theta,
    dml_equal = if (include_raw) raw$equal$theta else NA_real_,
    dml_middle = if (include_raw) raw$middle$theta else NA_real_,
    coding_difference = if (include_raw)
      raw$middle$theta - raw$equal$theta else NA_real_,
    max_abs_index = max(abs(full$index))
  )
  out <- c(out, full$diagnostics)
  if (!is.null(d$m0) && !is.null(d$g0) && !is.null(d$cuts) &&
      !is.null(d$theta)) {
    codings <- stage19_codings()
    out$oracle_dml_equal <- stage19_oracle_coding(d, codings$equal)
    out$oracle_dml_middle <- stage19_oracle_coding(d, codings$middle)
    oracle_d <- d$D - d$m0
    oracle_y <- d$latent - (d$theta * d$m0 + d$g0)
    out$oracle_latent_plr <- sum(oracle_d * oracle_y) / sum(oracle_d^2)
  }
  out
}

# The anchored-score estimators are Monte Carlo negative controls. Compute
# them on the original sample, but do not refit their nuisance models inside
# every bootstrap resample because no bootstrap inference targets them.
stage19_boot_point <- function(d, config, B = 2L, n_folds = 5L,
                               folds = NULL, level = 0.95) {
  original_sample <- !is.null(d$m0) && !is.null(d$g0) &&
    !is.null(d$cuts) && !is.null(d$theta)
  fit <- stage19_point(
    d, config, B = B, n_folds = n_folds, folds = folds, level = level,
    include_raw = original_sample
  )
  keep <- c("rao_blackwell", "direct_ordinal")
  fit$targets <- fit$targets[keep]
  fit$target_internal_se <- fit$target_internal_se[keep]
  fit
}

stage19_pipeline_boot <- function(
    d, config, B_outer = 99L, n_folds = 5L, level = 0.95,
    base_folds = NULL, bootstrap_indices = NULL, bootstrap_folds = NULL) {
  sudo_pipeline_boot(
    d, B_outer = B_outer, inner_B = 2L, level = level,
    n_folds = n_folds, estimator = stage19_boot_point, config = config,
    base_folds = base_folds,
    bootstrap_indices = bootstrap_indices,
    bootstrap_folds = bootstrap_folds
  )
}

stage19_save_bootstrap_state <- function(state, file) {
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile("stage19_state_", tmpdir = dirname(file))
  saveRDS(state, temporary)
  copied <- file.copy(temporary, file, overwrite = TRUE)
  unlink(temporary)
  if (!copied) stop("failed to save stage-19 bootstrap state")
  invisible(file)
}

stage19_pipeline_boot_resumable <- function(
    d, config, state_file, state_id, B_outer = 99L, n_folds = 5L,
    level = 0.95, base_folds = NULL, bootstrap_indices = NULL,
    bootstrap_folds = NULL) {
  stopifnot(
    is.character(state_file), length(state_file) == 1L,
    is.character(state_id), length(state_id) == 1L,
    B_outer >= 2L, n_folds >= 2L,
    length(bootstrap_indices) == B_outer,
    length(bootstrap_folds) == B_outer
  )
  state <- if (file.exists(state_file)) {
    try(readRDS(state_file), silent = TRUE)
  } else NULL
  valid_state <- is.list(state) && identical(state$id, state_id) &&
    identical(state$B_outer, as.integer(B_outer))
  if (!valid_state) {
    fit0 <- stage19_boot_point(
      d, config, n_folds = n_folds, folds = base_folds, level = level
    )
    target0 <- fit0$targets
    internal0 <- fit0$target_internal_se
    state <- list(
      id = state_id, B_outer = as.integer(B_outer), fit0 = fit0,
      target0 = target0, internal0 = internal0,
      boot_targets = matrix(
        NA_real_, nrow = length(target0), ncol = B_outer,
        dimnames = list(names(target0), NULL)
      ),
      boot_internal = matrix(
        NA_real_, nrow = length(target0), ncol = B_outer,
        dimnames = list(names(target0), NULL)
      ),
      thresholds_ordered = rep(NA_real_, B_outer),
      min_threshold_gap = rep(NA_real_, B_outer)
    )
    stage19_save_bootstrap_state(state, state_file)
  }
  fit0 <- state$fit0
  target0 <- state$target0
  remaining <- which(!is.finite(state$boot_targets[1L, ]))
  X <- as.data.frame(d$X)
  for (b in remaining) {
    index <- bootstrap_indices[[b]]
    fit <- stage19_boot_point(
      list(
        X = X[index, , drop = FALSE], D = d$D[index], Y = d$Y[index]
      ),
      config, n_folds = n_folds, folds = bootstrap_folds[[b]],
      level = level
    )
    if (!identical(names(fit$targets), names(target0))) {
      stop("bootstrap target names changed across stage-19 fits")
    }
    state$boot_targets[, b] <- fit$targets
    state$boot_internal[, b] <- fit$target_internal_se[names(target0)]
    state$thresholds_ordered[b] <- if (is.null(
      fit$thresholds_ordered
    )) 1 else fit$thresholds_ordered
    state$min_threshold_gap[b] <- if (is.null(
      fit$min_threshold_gap
    )) Inf else fit$min_threshold_gap
    stage19_save_bootstrap_state(state, state_file)
  }
  target_se <- apply(state$boot_targets, 1L, stats::sd)
  target_center <- rowMeans(state$boot_targets)
  z <- qnorm(1 - (1 - level) / 2)
  a <- (1 - level) / 2
  target_results <- data.frame(
    target = names(target0), estimate = as.numeric(target0),
    se = as.numeric(target_se),
    ci_lo = as.numeric(target0 - z * target_se),
    ci_hi = as.numeric(target0 + z * target_se),
    bootstrap_center = as.numeric(target_center),
    internal_se = as.numeric(state$internal0), row.names = NULL
  )
  primary <- names(target0)[1L]
  theta0 <- target0[[primary]]
  theta_boot <- state$boot_targets[primary, ]
  se <- target_se[[primary]]
  studentized <- (theta_boot - theta0) / state$boot_internal[primary, ]
  studentized_quantile <- stats::quantile(
    studentized[is.finite(studentized)], c(a, 1 - a)
  )
  out <- list(
    theta = theta0, se = se, ci_lo = theta0 - z * se,
    ci_hi = theta0 + z * se,
    ci_lo_pct = unname(stats::quantile(theta_boot, a)),
    ci_hi_pct = unname(stats::quantile(theta_boot, 1 - a)),
    ci_lo_stud = theta0 - unname(studentized_quantile[2L]) * fit0$se,
    ci_hi_stud = theta0 - unname(studentized_quantile[1L]) * fit0$se,
    internal_se = state$internal0[[primary]],
    bootstrap_center = mean(theta_boot), targets = target0,
    target_se = target_se, target_results = target_results,
    bootstrap_targets = state$boot_targets, B_outer = B_outer,
    fold_mode = "redraw", direct_theta = fit0$direct_theta,
    all_boot_thresholds_ordered = as.numeric(
      all(state$thresholds_ordered == 1)
    ),
    min_boot_threshold_gap = min(state$min_threshold_gap)
  )
  diagnostics <- setdiff(
    names(fit0), c(names(out), "ci_lo", "ci_hi", "theta", "se")
  )
  for (name in diagnostics) {
    value <- fit0[[name]]
    if (is.numeric(value) && length(value) == 1L) out[[name]] <- value
  }
  out
}

stage19_fit_binary_path <- function(d, folds, mstop_grid, nu = 0.1,
                                    knots = 10L, df = 4) {
  stopifnot(all(d$Y %in% 0:1), length(unique(d$Y)) == 2L)
  mstop_grid <- sort(unique(as.integer(mstop_grid)))
  stopifnot(all(mstop_grid > 0))
  X <- as.data.frame(d$X)
  dat <- data.frame(
    Y = factor(d$Y, levels = 0:1), D = d$D, X, check.names = FALSE
  )
  formula <- stage19_formula(names(X), TRUE, "Y", knots, df)
  n <- nrow(dat)
  output <- lapply(mstop_grid, function(step) list(
    index = numeric(n), lower = numeric(n), upper = numeric(n),
    direct = numeric(length(folds)), contrast_spread = numeric(length(folds)),
    probability_violation = 0
  ))
  names(output) <- as.character(mstop_grid)
  for (fold in seq_along(folds)) {
    test <- folds[[fold]]
    train <- setdiff(seq_len(n), test)
    if (length(unique(d$Y[train])) != 2L) {
      stop("a binary category is absent from a training fold")
    }
    maximum <- suppressWarnings(mboost::gamboost(
      formula, data = dat[train, , drop = FALSE],
      family = mboost::Binomial(type = "glm", link = "cloglog"),
      baselearner = mboost::bbs,
      control = mboost::boost_control(
        mstop = max(mstop_grid), nu = nu, trace = FALSE
      )
    ))
    for (step in mstop_grid) {
      model <- maximum[step]
      probability <- as.numeric(suppressWarnings(predict(
        model, newdata = dat[test, , drop = FALSE], type = "response"
      )))
      index <- as.numeric(suppressWarnings(predict(
        model, newdata = dat[test, , drop = FALSE], type = "link"
      )))
      treated <- dat[test, , drop = FALSE]
      control <- treated
      treated$D <- 1
      control$D <- 0
      contrast <- as.numeric(suppressWarnings(
        predict(model, newdata = treated, type = "link") -
          predict(model, newdata = control, type = "link")
      ))
      key <- as.character(step)
      output[[key]]$index[test] <- index
      output[[key]]$lower[test] <- ifelse(
        d$Y[test] == 0L, 0, 1 - probability
      )
      output[[key]]$upper[test] <- ifelse(
        d$Y[test] == 0L, 1 - probability, 1
      )
      output[[key]]$direct[fold] <- mean(contrast)
      output[[key]]$contrast_spread[fold] <- stats::sd(contrast)
      output[[key]]$probability_violation <- max(
        output[[key]]$probability_violation,
        max(pmax(-probability, probability - 1, 0))
      )
    }
  }
  lapply(output, function(value) list(
    index = value$index, lower = value$lower, upper = value$upper,
    reference = "gumbel_max", direct_theta = mean(value$direct),
    diagnostics = list(
      direct_theta_sd = stats::sd(value$direct),
      max_contrast_spread = max(value$contrast_spread),
      max_probability_violation = value$probability_violation,
      min_jump_width = min(value$upper - value$lower)
    )
  ))
}

stage19_binary_theta_from_full <- function(d, folds, full, config, B = 3L) {
  stopifnot(B >= 2L)
  .check_discrete_fit(full, length(d$Y))
  X <- as.data.frame(d$X)
  treatment_fitter <- stage19_nuisance_fitter("treatment", config)
  outcome_fitter <- stage19_nuisance_fitter("outcome", config)
  D_res <- d$D - crossfit(X, d$D, treatment_fitter, folds)
  draws <- vapply(seq_len(B), function(draw) {
    surrogate <- complete_discrete(
      full$lower, full$upper, full$index, full$reference
    )
    nuisance <- crossfit(X, surrogate, outcome_fitter, folds)
    estimate <- fwl_theta(surrogate - nuisance, D_res)
    c(theta = estimate$theta, var = estimate$var)
  }, numeric(2))
  pooled <- pool_rubin(
    draws["theta", ], draws["var", ], n_obs = length(d$Y)
  )
  list(theta = pooled$theta, se = pooled$se, draws = draws)
}

stage19_unit_checks <- function() {
  stage19_require_packages()
  intervals <- rbind(c(0, 0.2), c(0.2, 0.8), c(0.8, 1))
  analytic <- stage19_logistic_jump_mean(intervals[, 1], intervals[, 2])
  numeric <- apply(intervals, 1L, function(bounds) integrate(
    function(u) qlogis(pmin(pmax(u, 1e-12), 1 - 1e-12)),
    bounds[1], bounds[2], rel.tol = 1e-9
  )$value / diff(bounds))
  stopifnot(max(abs(analytic - numeric)) < 1e-6)

  set.seed(19001L)
  d <- stage19_dgp(400L, 8L, 1)
  stopifnot(length(table(d$Y)) == 5L, min(prop.table(table(d$Y))) > 0.05)
  probability <- stage19_category_probabilities(d$eta, d$cuts)
  stopifnot(
    all(probability >= 0), max(abs(rowSums(probability) - 1)) < 1e-12
  )
  folds <- stage19_make_folds(400L, 2L, 19002L)
  config <- stage19_config(
    "pure", mstop = 100L, nuisance_mstop = 100L, knots = 8L
  )
  fit <- stage19_point(
    d, config, n_folds = 2L, folds = folds, include_raw = FALSE
  )
  stopifnot(
    all(is.finite(fit$targets)), fit$thresholds_ordered == 1,
    fit$min_threshold_gap > 0, fit$max_contrast_spread < 1e-8,
    fit$max_probability_error < 1e-6
  )

  set.seed(19003L)
  binary <- stage19_binary_dgp(400L, 8L, 1)
  binary_folds <- stage19_make_folds(400L, 2L, 19004L)
  binary_fit <- stage19_fit_binary_path(
    binary, binary_folds, 100L, knots = 8L
  )[[1L]]
  .check_discrete_fit(binary_fit, 400L)
  binary_surrogate <- complete_discrete(
    binary_fit$lower, binary_fit$upper, binary_fit$index,
    binary_fit$reference
  )
  stopifnot(
    all(is.finite(binary_surrogate)),
    binary_fit$diagnostics$max_probability_violation < 1e-12,
    binary_fit$diagnostics$max_contrast_spread < 1e-8,
    binary_fit$diagnostics$min_jump_width > 0
  )
  invisible(TRUE)
}
