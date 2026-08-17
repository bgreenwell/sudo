# Experimental support for stage 16. These functions stay outside R/sudo/
# until the full R validation gates pass.

stage16_require_packages <- function() {
  if (!requireNamespace("gamsel", quietly = TRUE)) {
    stop("stage 16 requires the CRAN package 'gamsel'")
  }
}

stage16_env_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}

stage16_env_num <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.numeric(value) else as.numeric(default)
}

stage16_entropy <- function(probability, clip = 1e-12) {
  probability <- pmin(pmax(probability, clip), 1 - clip)
  -probability * log(probability) -
    (1 - probability) * log1p(-probability)
}

stage16_rb_completion <- function(y, index) {
  probability <- plogis(index)
  entropy <- stage16_entropy(probability)
  index + ifelse(
    y == 1L,
    entropy / pmax(probability, 1e-12),
    -entropy / pmax(1 - probability, 1e-12)
  )
}

stage16_orthogonal_completion <- function(y, index, clip) {
  probability_raw <- plogis(index)
  probability <- pmin(pmax(probability_raw, clip), 1 - clip)
  value <- index + (y - probability) /
    (probability * (1 - probability))
  list(
    value = value,
    probability = probability,
    clipping_rate = mean(probability != probability_raw),
    max_inverse_information = max(1 / (probability * (1 - probability)))
  )
}

stage16_cells <- function(smoke = FALSE) {
  cells <- data.frame(
    cell = c("randomized", "confounded", "high_signal", "high_dimensional"),
    design = c("randomized", "confounded", "high_signal", "randomized"),
    n = c(3000L, 3000L, 3000L, 2000L),
    p = c(80L, 80L, 80L, 500L),
    primary = c(TRUE, TRUE, FALSE, FALSE),
    stringsAsFactors = FALSE
  )
  if (smoke) {
    cells$n <- c(350L, 350L, 350L, 300L)
    cells$p <- c(12L, 12L, 12L, 24L)
  }
  cells
}

stage16_dgp <- function(n, p, design, theta = 0.75) {
  stopifnot(p >= 6L)
  X <- matrix(rnorm(n * p), n, p)
  colnames(X) <- paste0("X", seq_len(p))
  g0 <- 0.65 * X[, 1] + 0.55 * (X[, 2]^2 - 1) +
    0.45 * X[, 3] - 0.35 * (X[, 4]^2 - 1) +
    0.30 * X[, 5] - 0.25 * X[, 6]
  m0 <- if (design == "confounded") {
    0.50 * X[, 1] - 0.40 * X[, 3] + 0.30 * (X[, 2]^2 - 1)
  } else {
    rep(0, n)
  }
  noise_sd <- if (design == "high_signal") 2 else 1
  D <- m0 + noise_sd * rnorm(n)
  eta <- theta * D + g0
  Y <- rbinom(n, 1, plogis(eta))
  list(
    X = X, D = D, Y = Y, theta = theta, g0 = g0, m0 = m0,
    eta = eta, design = design, noise_sd = noise_sd
  )
}

stage16_fold_id <- function(n, folds, seed) {
  set.seed(seed)
  sample(rep_len(seq_len(folds), n))
}

stage16_cv_fit <- function(x, y, family, foldid, treatment_column = FALSE,
                           gamma = 0.4) {
  x <- as.matrix(x)
  p <- ncol(x)
  degrees <- rep(3L, p)
  dfs <- rep(3, p)
  if (treatment_column) {
    degrees[1] <- 1L
    dfs[1] <- 1
  }
  bases <- gamsel::pseudo.bases(x, degree = degrees, df = dfs)
  gamsel::cv.gamsel(
    x, y, family = family, degrees = degrees, dfs = dfs, bases = bases,
    foldid = foldid, gamma = gamma,
    type.measure = if (family == "binomial") "deviance" else "mse"
  )
}

stage16_predict <- function(fit, newdata, selection = c("min", "one_se")) {
  selection <- match.arg(selection)
  index <- if (selection == "min") fit$index.min else fit$index.1se
  as.numeric(predict(
    fit$gamsel.fit, newdata = as.matrix(newdata), type = "link",
    index = index
  ))
}

stage16_predict_response <- function(fit, newdata,
                                     selection = c("min", "one_se")) {
  selection <- match.arg(selection)
  index <- if (selection == "min") fit$index.min else fit$index.1se
  as.numeric(predict(
    fit$gamsel.fit, newdata = as.matrix(newdata), type = "response",
    index = index
  ))
}

stage16_safe_root <- function(fn, lower = -3, upper = 3) {
  left <- fn(lower)
  right <- fn(upper)
  for (iteration in seq_len(4L)) {
    if (is.finite(left) && is.finite(right) && left * right <= 0) {
      return(stats::uniroot(fn, c(lower, upper), tol = 1e-9)$root)
    }
    lower <- lower * 2
    upper <- upper * 2
    left <- fn(lower)
    right <- fn(upper)
  }
  NA_real_
}

stage16_score_calibration <- function(
    y, index, direction, direct,
    mode = c("full", "scale", "tilt")) {
  mode <- match.arg(mode)
  design <- switch(
    mode,
    full = cbind(intercept = 1, index = index, direction = direction),
    scale = cbind(intercept = 1, index = index),
    tilt = cbind(intercept = 1, direction = direction)
  )
  offset <- if (mode == "tilt") index else NULL
  fit <- try(suppressWarnings(stats::glm.fit(
    x = design, y = y, family = stats::binomial(), offset = offset,
    control = stats::glm.control(epsilon = 1e-10, maxit = 100L)
  )), silent = TRUE)
  if (inherits(fit, "try-error") || !isTRUE(fit$converged) ||
      any(!is.finite(fit$coefficients))) {
    return(c(
      theta = NA_real_, intercept = NA_real_, slope = NA_real_,
      tilt = NA_real_, score_error = NA_real_, converged = 0
    ))
  }
  coefficient <- fit$coefficients
  slope <- if (mode == "tilt") 1 else coefficient["index"]
  tilt <- if (mode == "scale") 0 else coefficient["direction"]
  theta <- slope * mean(direct) + tilt
  score_error <- max(abs(crossprod(
    design, y - fit$fitted.values
  ))) / length(y)
  c(
    theta = unname(theta),
    intercept = unname(coefficient["intercept"]),
    slope = unname(slope), tilt = unname(tilt),
    score_error = unname(score_error), converged = 1
  )
}

stage16_pit_calibration <- function(y, index, direction, direct,
                                    affine = FALSE) {
  stage16_score_calibration(
    y, index, direction, direct,
    mode = if (affine) "full" else "tilt"
  )
}

stage16_weighted_center <- function(index, alpha, treatment, treatment_mean,
                                    treatment_sd, weight = c("information",
                                                             "entropy")) {
  weight <- match.arg(weight)
  grid_standard <- qnorm((seq_len(81L) - 0.5) / 81L)
  f_part <- index - alpha * treatment
  grid <- outer(treatment_mean, rep(1, length(grid_standard))) +
    outer(treatment_sd, grid_standard)
  grid_index <- outer(f_part, rep(1, length(grid_standard))) +
    sweep(grid, 1L, alpha, "*")
  probability <- plogis(grid_index)
  weights <- if (weight == "information") {
    probability * (1 - probability)
  } else {
    stage16_entropy(probability)
  }
  as.numeric(rowSums(weights * grid) / pmax(rowSums(weights), 1e-12))
}

stage16_crossfit_affine_sudo <- function(
    y, index, direction, direct, x, residual_d, outer_id, inner_folds,
    seed, calibration_mode = "full") {
  calibration_mode <- match.arg(
    calibration_mode, c("full", "scale", "tilt")
  )
  n <- length(y)
  calibrated_index <- rep(NA_real_, n)
  completed <- rep(NA_real_, n)
  nuisance <- rep(NA_real_, n)
  calibration <- matrix(
    NA_real_, max(outer_id), 5L,
    dimnames = list(NULL, c(
      "intercept", "slope", "tilt", "score_error", "converged"
    ))
  )
  for (fold in seq_len(max(outer_id))) {
    test <- which(outer_id == fold)
    train <- which(outer_id != fold)
    fit <- stage16_score_calibration(
      y[train], index[train], direction[train], direct[train],
      mode = calibration_mode
    )
    calibration[fold, ] <- fit[c(
      "intercept", "slope", "tilt", "score_error", "converged"
    )]
    if (!isTRUE(unname(fit["converged"]) == 1) ||
        any(!is.finite(fit[c("intercept", "slope", "tilt")]))) {
      next
    }
    train_index <- unname(fit["intercept"]) +
      unname(fit["slope"]) * index[train] +
      unname(fit["tilt"]) * direction[train]
    test_index <- unname(fit["intercept"]) +
      unname(fit["slope"]) * index[test] +
      unname(fit["tilt"]) * direction[test]
    train_completed <- stage16_rb_completion(y[train], train_index)
    inner_id <- stage16_fold_id(
      length(train), inner_folds, seed + 1000L + fold
    )
    nuisance_fit <- stage16_cv_fit(
      x[train, , drop = FALSE], train_completed, "gaussian", inner_id
    )
    calibrated_index[test] <- test_index
    completed[test] <- stage16_rb_completion(y[test], test_index)
    nuisance[test] <- stage16_predict(
      nuisance_fit, x[test, , drop = FALSE], "min"
    )
  }
  denominator <- sum(residual_d^2)
  valid <- all(is.finite(completed)) && all(is.finite(nuisance)) &&
    is.finite(denominator) && denominator > 0
  list(
    theta = if (valid) {
      sum(residual_d * (completed - nuisance)) / denominator
    } else {
      NA_real_
    },
    index = calibrated_index,
    completed = completed,
    nuisance = nuisance,
    mean_intercept = mean(calibration[, "intercept"], na.rm = TRUE),
    mean_slope = mean(calibration[, "slope"], na.rm = TRUE),
    mean_tilt = mean(calibration[, "tilt"], na.rm = TRUE),
    max_score_error = max(calibration[, "score_error"], na.rm = TRUE),
    convergence_rate = mean(calibration[, "converged"] == 1, na.rm = TRUE)
  )
}

stage16_fit_pipeline <- function(data, clip = 1e-6, outer_folds = 5L,
                                 inner_folds = 5L, seed = 1L,
                                 include_one_se = TRUE,
                                 include_calibration = FALSE,
                                 include_calibration_ablation = FALSE) {
  stage16_require_packages()
  X <- as.matrix(data$X)
  D <- as.numeric(data$D)
  Y <- as.integer(data$Y)
  n <- nrow(X)
  selections <- if (include_one_se) c("min", "one_se") else "min"
  outer_id <- stage16_fold_id(n, outer_folds, seed + 11L)
  fold_signature <- sum(seq_len(n) * outer_id)
  storage <- setNames(lapply(selections, function(selection) list(
    index = numeric(n), direct = numeric(n), rb = numeric(n),
    orthogonal = numeric(n), l_index = numeric(n), l_rb = numeric(n),
    l_orthogonal = numeric(n), treatment_mean = numeric(n),
    treatment_sd = numeric(n), clipping = numeric(n),
    max_inverse_information = numeric(n),
    direct_contrast_spread = numeric(n)
  )), selections)
  m_hat <- numeric(n)
  raw_outcome_hat <- numeric(n)

  for (fold in seq_len(outer_folds)) {
    test <- which(outer_id == fold)
    train <- which(outer_id != fold)
    inner_id <- stage16_fold_id(
      length(train), inner_folds, seed + 1000L + fold
    )
    x_train <- X[train, , drop = FALSE]
    x_test <- X[test, , drop = FALSE]
    z_train <- cbind(D = D[train], x_train)
    z_test <- cbind(D = D[test], x_test)
    full_fit <- stage16_cv_fit(
      z_train, Y[train], "binomial", inner_id, treatment_column = TRUE
    )
    treatment_fit <- stage16_cv_fit(
      x_train, D[train], "gaussian", inner_id
    )
    raw_outcome_fit <- stage16_cv_fit(
      x_train, Y[train], "binomial", inner_id
    )
    m_test <- stage16_predict(treatment_fit, x_test, "min")
    m_train <- stage16_predict(treatment_fit, x_train, "min")
    treatment_sd <- stats::sd(D[train] - m_train)
    m_hat[test] <- m_test
    raw_outcome_hat[test] <- stage16_predict_response(
      raw_outcome_fit, x_test, "min"
    )

    for (selection in selections) {
      v_train <- stage16_predict(full_fit, z_train, selection)
      v_test <- stage16_predict(full_fit, z_test, selection)
      z_plus <- z_test
      z_plus[, 1] <- z_plus[, 1] + 1
      direct <- stage16_predict(full_fit, z_plus, selection) - v_test
      rb_train <- stage16_rb_completion(Y[train], v_train)
      rb_test <- stage16_rb_completion(Y[test], v_test)
      orth_train <- stage16_orthogonal_completion(Y[train], v_train, clip)
      orth_test <- stage16_orthogonal_completion(Y[test], v_test, clip)
      index_fit <- stage16_cv_fit(
        x_train, v_train, "gaussian", inner_id
      )
      rb_fit <- stage16_cv_fit(
        x_train, rb_train, "gaussian", inner_id
      )
      orth_fit <- stage16_cv_fit(
        x_train, orth_train$value, "gaussian", inner_id
      )
      current <- storage[[selection]]
      current$index[test] <- v_test
      current$direct[test] <- direct
      current$rb[test] <- rb_test
      current$orthogonal[test] <- orth_test$value
      current$l_index[test] <- stage16_predict(index_fit, x_test, "min")
      current$l_rb[test] <- stage16_predict(rb_fit, x_test, "min")
      current$l_orthogonal[test] <- stage16_predict(
        orth_fit, x_test, "min"
      )
      current$treatment_mean[test] <- m_test
      current$treatment_sd[test] <- treatment_sd
      current$clipping[test] <- as.numeric(
        plogis(v_test) < clip | plogis(v_test) > 1 - clip
      )
      current$max_inverse_information[test] <-
        1 / (orth_test$probability * (1 - orth_test$probability))
      current$direct_contrast_spread[test] <- diff(range(direct))
      storage[[selection]] <- current
    }
  }

  residual_d <- D - m_hat
  denominator <- sum(residual_d^2)
  if (!is.finite(denominator) || denominator <= 0) {
    stop("invalid treatment residual denominator")
  }
  estimates <- list()
  diagnostics <- list()
  for (selection in selections) {
    current <- storage[[selection]]
    index_projection <- sum(
      residual_d * (current$index - current$l_index)
    ) / denominator
    rb_theta <- sum(residual_d * (current$rb - current$l_rb)) / denominator
    orthogonal_theta <- sum(
      residual_d * (current$orthogonal - current$l_orthogonal)
    ) / denominator
    correction <- rb_theta - index_projection
    probability <- plogis(current$index)
    entropy <- stage16_entropy(probability)
    kappa <- sum(residual_d^2 * (1 - entropy)) / denominator
    scaled <- index_projection + correction / pmax(1 - kappa, 1e-8)
    information_center <- stage16_weighted_center(
      current$index, current$direct, D, current$treatment_mean,
      current$treatment_sd, "information"
    )
    h_information <- D - information_center
    targeted_root <- stage16_safe_root(function(epsilon) mean(
      h_information *
        (Y - plogis(current$index + epsilon * h_information))
    ))
    targeted_projection <- mean(current$direct) + targeted_root
    targeted_index <- current$index + targeted_root * h_information
    targeted_rb <- stage16_rb_completion(Y, targeted_index)
    targeted_correction <- sum(
      residual_d * (targeted_rb - targeted_index)
    ) / denominator
    targeted_sudo <- targeted_projection + targeted_correction
    estimate <- c(
      direct = mean(current$direct),
      raw_label = sum(residual_d * (Y - raw_outcome_hat)) / denominator,
      index_projection = index_projection,
      sudo = rb_theta,
      scaled = scaled,
      orthogonal = orthogonal_theta,
      targeted_projection = targeted_projection,
      targeted_sudo = targeted_sudo
    )
    diagnostic <- c(
      correction = correction,
      kappa = kappa,
      clipping_rate = mean(current$clipping),
      max_inverse_information = max(current$max_inverse_information),
      direct_contrast_spread = max(current$direct_contrast_spread),
      target_failure = as.numeric(!is.finite(targeted_root)),
      index_rmse = if (!is.null(data$eta))
        sqrt(mean((current$index - data$eta)^2)) else NA_real_,
      index_correlation = if (!is.null(data$eta))
        stats::cor(current$index, data$eta) else NA_real_
    )
    if (include_calibration) {
      intercept_tilt <- stage16_pit_calibration(
        Y, current$index, h_information, current$direct, affine = FALSE
      )
      affine_tilt <- stage16_pit_calibration(
        Y, current$index, h_information, current$direct, affine = TRUE
      )
      estimate <- c(
        estimate,
        pit_intercept_tilt = unname(intercept_tilt["theta"]),
        pit_affine_tilt = unname(affine_tilt["theta"])
      )
      affine_sudo <- stage16_crossfit_affine_sudo(
        Y, current$index, h_information, current$direct, X, residual_d,
        outer_id, inner_folds, seed + 50000L
      )
      estimate <- c(
        estimate,
        affine_calibrated_sudo = unname(affine_sudo$theta)
      )
      names(intercept_tilt) <- paste0(
        "pit_intercept_tilt_", names(intercept_tilt)
      )
      names(affine_tilt) <- paste0("pit_affine_tilt_", names(affine_tilt))
      affine_sudo_diagnostic <- c(
        mean_intercept = affine_sudo$mean_intercept,
        mean_slope = affine_sudo$mean_slope,
        mean_tilt = affine_sudo$mean_tilt,
        max_score_error = affine_sudo$max_score_error,
        convergence_rate = affine_sudo$convergence_rate
      )
      names(affine_sudo_diagnostic) <- paste0(
        "affine_calibrated_sudo_", names(affine_sudo_diagnostic)
      )
      diagnostic <- c(
        diagnostic, intercept_tilt, affine_tilt, affine_sudo_diagnostic
      )
      if (include_calibration_ablation) {
        affine_scale <- stage16_score_calibration(
          Y, current$index, h_information, current$direct, mode = "scale"
        )
        scale_sudo <- stage16_crossfit_affine_sudo(
          Y, current$index, h_information, current$direct, X, residual_d,
          outer_id, inner_folds, seed + 60000L, calibration_mode = "scale"
        )
        tilt_sudo <- stage16_crossfit_affine_sudo(
          Y, current$index, h_information, current$direct, X, residual_d,
          outer_id, inner_folds, seed + 70000L, calibration_mode = "tilt"
        )
        estimate <- c(
          estimate,
          pit_affine_scale = unname(affine_scale["theta"]),
          scale_calibrated_sudo = unname(scale_sudo$theta),
          tilt_calibrated_sudo = unname(tilt_sudo$theta)
        )
        names(affine_scale) <- paste0(
          "pit_affine_scale_", names(affine_scale)
        )
        diagnostic <- c(diagnostic, affine_scale)
        for (mode in c("scale", "tilt")) {
          value <- if (mode == "scale") scale_sudo else tilt_sudo
          mode_diagnostic <- c(
            mean_intercept = value$mean_intercept,
            mean_slope = value$mean_slope,
            mean_tilt = value$mean_tilt,
            max_score_error = value$max_score_error,
            convergence_rate = value$convergence_rate
          )
          names(mode_diagnostic) <- paste0(
            mode, "_calibrated_sudo_", names(mode_diagnostic)
          )
          diagnostic <- c(diagnostic, mode_diagnostic)
        }
      }
    }
    estimates[[selection]] <- estimate
    diagnostics[[selection]] <- diagnostic
  }
  list(
    estimates = estimates, diagnostics = diagnostics,
    fold_signature = fold_signature,
    sample_signature = sum(Y * seq_along(Y)),
    m_hat = m_hat
  )
}

stage16_oracle <- function(data, clip = 1e-6) {
  residual_d <- data$D - data$m0
  denominator <- sum(residual_d^2)
  rb <- stage16_rb_completion(data$Y, data$eta)
  orthogonal <- stage16_orthogonal_completion(data$Y, data$eta, clip)
  nuisance <- data$theta * data$m0 + data$g0
  c(
    oracle_sudo = sum(residual_d * (rb - nuisance)) / denominator,
    oracle_orthogonal = sum(
      residual_d * (orthogonal$value - nuisance)
    ) / denominator
  )
}

stage16_one_rep <- function(seed, cell, clip, outer_folds, inner_folds,
                            include_one_se = TRUE,
                            include_calibration = FALSE,
                            include_calibration_ablation = FALSE) {
  set.seed(seed)
  data <- stage16_dgp(cell$n, cell$p, cell$design, theta = 0.75)
  fit <- stage16_fit_pipeline(
    data, clip, outer_folds, inner_folds, seed + 100000L,
    include_one_se, include_calibration, include_calibration_ablation
  )
  oracle <- stage16_oracle(data, clip)
  rows <- lapply(names(fit$estimates), function(selection) {
    data.frame(
      seed = seed, selection = selection, truth = data$theta,
      as.list(fit$estimates[[selection]]),
      as.list(fit$diagnostics[[selection]]),
      oracle_sudo = oracle["oracle_sudo"],
      oracle_orthogonal = oracle["oracle_orthogonal"],
      sample_signature = fit$sample_signature,
      fold_signature = fit$fold_signature,
      stringsAsFactors = FALSE, check.names = FALSE
    )
  })
  do.call(rbind, rows)
}

stage16_summarize_point <- function(replications) {
  estimators <- c(
    "direct", "raw_label", "index_projection", "sudo", "scaled", "orthogonal",
    "targeted_projection", "targeted_sudo", "oracle_sudo",
    "oracle_orthogonal"
  )
  blocks <- split(
    replications,
    list(replications$cell, replications$selection), drop = TRUE
  )
  rows <- list()
  position <- 0L
  for (block in blocks) {
    for (estimator in estimators) {
      value <- block[[estimator]]
      ok <- is.finite(value)
      value <- value[ok]
      truth <- block$truth[ok]
      position <- position + 1L
      rows[[position]] <- data.frame(
        cell = block$cell[1], selection = block$selection[1],
        primary = block$primary[1], estimator = estimator,
        n_reps = length(value), mean_est = mean(value),
        bias = mean(value - truth),
        mc_se_bias = stats::sd(value - truth) / sqrt(length(value)),
        sd = stats::sd(value),
        rmse = sqrt(mean((value - truth)^2)),
        fail_rate = mean(!ok), stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

stage16_point_gates <- function(summary) {
  primary <- summary$primary & summary$selection == "min" &
    summary$estimator == "orthogonal"
  bias_ok <- abs(summary$bias) <= pmax(
    2 * summary$mc_se_bias, 0.03 * 0.75
  )
  ordinary <- summary[
    summary$primary & summary$selection == "min" &
      summary$estimator == "sudo", c("cell", "bias")
  ]
  improvement <- merge(
    summary[primary, c("cell", "bias")], ordinary,
    by = "cell", suffixes = c("_orthogonal", "_sudo")
  )
  informative <- summary[
    summary$primary & summary$selection == "min" &
      summary$estimator == "direct", , drop = FALSE
  ]
  list(
    primary_pass = all(bias_ok[primary] & summary$fail_rate[primary] < 0.02),
    improvement_pass = all(
      abs(improvement$bias_orthogonal) <= abs(improvement$bias_sudo)
    ),
    informativeness_pass = any(abs(informative$bias) >= 0.05),
    bias_ok = bias_ok
  )
}

stage16_bootstrap_rep <- function(seed, cell, clip, outer_folds, inner_folds,
                                  bootstrap_reps,
                                  fit_timeout_seconds = 600L,
                                  include_calibration = FALSE) {
  set.seed(seed)
  data <- stage16_dgp(cell$n, cell$p, cell$design, theta = 0.75)
  point <- stage16_fit_pipeline(
    data, clip, outer_folds, inner_folds, seed + 200000L,
    include_one_se = FALSE,
    include_calibration = include_calibration
  )
  point_estimates <- point$estimates$min
  bootstrap <- matrix(
    NA_real_, bootstrap_reps, length(point_estimates),
    dimnames = list(NULL, names(point_estimates))
  )
  set.seed(seed + 300000L)
  indices <- replicate(bootstrap_reps, sample.int(cell$n, replace = TRUE),
                       simplify = FALSE)
  for (b in seq_len(bootstrap_reps)) {
    index <- indices[[b]]
    boot_data <- list(
      X = data$X[index, , drop = FALSE], D = data$D[index],
      Y = data$Y[index]
    )
    setTimeLimit(elapsed = fit_timeout_seconds, transient = TRUE)
    fit <- try(stage16_fit_pipeline(
      boot_data, clip, outer_folds, inner_folds,
      seed + 400000L + b, include_one_se = FALSE,
      include_calibration = include_calibration
    ), silent = TRUE)
    setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
    if (!inherits(fit, "try-error")) {
      bootstrap[b, ] <- fit$estimates$min[names(point_estimates)]
    }
  }
  rows <- lapply(names(point_estimates), function(estimator) {
    values <- bootstrap[, estimator]
    values <- values[is.finite(values)]
    standard_error <- stats::sd(values)
    estimate <- point_estimates[estimator]
    data.frame(
      estimator = estimator, truth = data$theta, estimate = estimate,
      se = standard_error,
      covered = abs(estimate - data$theta) <= qnorm(0.975) * standard_error,
      bootstrap_success = length(values) / bootstrap_reps
    )
  })
  do.call(rbind, rows)
}

stage16_summarize_coverage <- function(replications) {
  blocks <- split(
    replications,
    list(replications$cell, replications$estimator), drop = TRUE
  )
  rows <- lapply(blocks, function(block) {
    n <- nrow(block)
    sd_value <- stats::sd(block$estimate)
    mean_se <- mean(block$se)
    coverage <- mean(block$covered)
    ratio <- sd_value / mean_se
    ratio_mc_se <- ratio * sqrt(1 / (2 * (n - 1)))
    data.frame(
      cell = block$cell[1], primary = block$primary[1],
      estimator = block$estimator[1], n_reps = n,
      bias = mean(block$estimate - block$truth),
      mc_se_bias = sd_value / sqrt(n), sd = sd_value, mean_se = mean_se,
      sd_to_se = ratio, ratio_mc_se = ratio_mc_se,
      coverage = coverage,
      mc_se_coverage = sqrt(coverage * (1 - coverage) / n),
      mean_bootstrap_success = mean(block$bootstrap_success)
    )
  })
  do.call(rbind, rows)
}

stage16_coverage_gates <- function(summary) {
  primary <- summary$primary & summary$estimator == "orthogonal"
  bias_ok <- abs(summary$bias) <= pmax(
    2 * summary$mc_se_bias, 0.03 * 0.75
  )
  coverage_ok <- abs(summary$coverage - 0.95) <= pmax(
    2 * summary$mc_se_coverage, 0.04
  )
  ratio_ok <- abs(summary$sd_to_se - 1) <= 2 * summary$ratio_mc_se
  list(
    pass = all(
      bias_ok[primary] & coverage_ok[primary] & ratio_ok[primary] &
        summary$mean_bootstrap_success[primary] >= 0.98
    ),
    bias_ok = bias_ok, coverage_ok = coverage_ok, ratio_ok = ratio_ok
  )
}

stage16_cluster <- function(cores) {
  cluster <- parallel::makeCluster(cores)
  parallel::clusterEvalQ(cluster, {
    source("R/stage16_orthogonal_support.R")
    suppressPackageStartupMessages(library(gamsel))
    NULL
  })
  cluster
}

stage16_unit_checks <- function() {
  stopifnot(identical(
    stage16_fold_id(40L, 5L, 160000L),
    stage16_fold_id(40L, 5L, 160000L)
  ))
  index <- c(-8, -2, 0, 2, 8)
  y <- c(0L, 1L, 0L, 1L, 1L)
  probability <- plogis(index)
  information <- probability * (1 - probability)
  entropy <- stage16_entropy(probability)
  rb <- stage16_rb_completion(y, index)
  completed <- rb + (1 - entropy) / information * (y - probability)
  working <- index + (y - probability) / information
  stopifnot(max(abs(completed - working)) < 1e-10)
  extreme <- stage16_orthogonal_completion(
    c(0L, 1L), c(-40, 40), clip = 1e-4
  )
  stopifnot(all(is.finite(extreme$value)), extreme$clipping_rate == 1)

  set.seed(160001L)
  n <- 200000L
  index0 <- rnorm(n)
  y0 <- rbinom(n, 1, plogis(index0))
  direction <- rnorm(n)
  moment <- function(step) {
    shifted <- index0 + step * direction
    probability <- plogis(shifted)
    information <- probability * (1 - probability)
    rb_value <- stage16_rb_completion(y0, shifted)
    augmentation <- (1 - stage16_entropy(probability)) /
      information * (y0 - probability)
    mean(direction * (rb_value + augmentation))
  }
  derivative <- (moment(1e-4) - moment(-1e-4)) / 2e-4
  stopifnot(abs(derivative) < 0.03)

  set.seed(160002L)
  treatment <- rnorm(2000L)
  true_index <- 0.75 * treatment + rnorm(2000L)
  outcome <- rbinom(2000L, 1L, plogis(true_index))
  calibration <- stage16_pit_calibration(
    outcome, 0.8 * true_index, treatment, rep(0.6, 2000L), affine = TRUE
  )
  stopifnot(calibration["converged"] == 1,
            calibration["score_error"] < 1e-8,
            all(is.finite(calibration)))
  invisible(TRUE)
}
