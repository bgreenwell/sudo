# Uniform adapter layer for discrete-outcome SUDO.
#
# Each adapter returns an object with the following fields:
#   index          fitted scalar index eta(D, X); no coefficient is required
#   lower, upper   fitted randomized-PIT CDF bounds for the observed Y
#   reference      mean-zero reference law used by complete_discrete()
#   direct_theta   optional coefficient from a structured comparator
#   draw()         optional proper parameter draw returning new bounds/index
#   diagnostics    optional scalar or vector fit diagnostics
#
# The flexible adapters cross-fit the full distribution model. Their
# uncertainty is propagated by sudo_pipeline_boot(); within one pipeline fit,
# repeated completions randomize only the PIT. This is the frequentist
# counterpart of the existing binary flexible-learner implementation.

.check_discrete_fit <- function(fit, n) {
  required <- c("index", "lower", "upper", "reference")
  missing <- setdiff(required, names(fit))
  if (length(missing)) {
    stop("discrete adapter omitted: ", paste(missing, collapse = ", "))
  }
  stopifnot(length(fit$index) == n, length(fit$lower) == n,
            length(fit$upper) == n, all(is.finite(fit$index)),
            all(is.finite(fit$lower)), all(is.finite(fit$upper)),
            all(fit$lower >= 0), all(fit$upper <= 1),
            all(fit$lower <= fit$upper))
  invisible(fit)
}

.fit_draw_spec <- function(fit) {
  out <- if (is.function(fit$draw)) fit$draw() else fit
  if (is.null(out$reference)) out$reference <- fit$reference
  out
}

sudo_discrete <- function(d, adapter, B = 25L, n_folds = 5L, folds = NULL,
                          refit_S_nuisance = TRUE, fit_l = fit_gam,
                          fit_m = NULL, level = 0.95) {
  stopifnot(is.function(adapter), B >= 2L, n_folds >= 2L)
  X <- as.data.frame(d$X)
  n <- nrow(X)
  stopifnot(length(d$Y) == n, length(d$D) == n)
  if (is.null(folds)) folds <- make_folds(n, n_folds)
  stopifnot(is.list(folds), length(folds) == n_folds,
            identical(sort(as.integer(unlist(folds))), seq_len(n)))
  if (is.null(fit_m)) {
    is_binary_D <- all(d$D %in% c(0, 1))
    fit_m <- if (is_binary_D) fit_gam_binomial else fit_gam
  }

  full <- adapter(d, folds)
  .check_discrete_fit(full, n)
  D_res <- d$D - crossfit(X, d$D, fit_m, folds)

  initial <- complete_discrete(full$lower, full$upper, full$index,
                               full$reference)
  ell_initial <- if (refit_S_nuisance) NULL else
    crossfit(X, initial, fit_l, folds)

  draws <- vapply(seq_len(B), function(b) {
    spec <- .fit_draw_spec(full)
    .check_discrete_fit(spec, n)
    surrogate <- complete_discrete(spec$lower, spec$upper, spec$index,
                                    spec$reference)
    ell <- if (refit_S_nuisance) crossfit(X, surrogate, fit_l, folds) else
      ell_initial
    fwl <- fwl_theta(surrogate - ell, D_res)
    c(theta = fwl$theta, var = fwl$var)
  }, numeric(2))
  pooled <- pool_rubin(draws["theta", ], draws["var", ], level = level,
                       n_obs = n)
  out <- list(
    theta = pooled$theta,
    se = pooled$se,
    ci_lo = pooled$ci_lo,
    ci_hi = pooled$ci_hi,
    W = pooled$W,
    B_between = pooled$B_between,
    df = pooled$df,
    direct_theta = if (is.null(full$direct_theta)) NA_real_ else
      full$direct_theta,
    max_abs_index = max(abs(full$index))
  )
  if (!is.null(full$diagnostics)) out <- c(out, full$diagnostics)
  out
}

.binary_link <- function(link) {
  link <- match.arg(link, c("logit", "probit", "cloglog"))
  list(
    family = stats::binomial(link = link),
    inverse = switch(link, logit = plogis, probit = pnorm,
                     cloglog = function(x) -expm1(-exp(pmin(x, 30)))),
    reference = switch(link, logit = "logistic", probit = "normal",
                       cloglog = "gumbel_max")
  )
}

# Link-aware partially-linear binary GAM. D is linear and every X term is
# smooth, so the model always supplies a scalar additive treatment index.
fit_binary_pl_gam_adapter <- function(d, folds, link = "logit",
                                      method = "REML") {
  law <- .binary_link(link)
  X <- as.data.frame(d$X)
  dat <- data.frame(Y = d$Y, D = d$D, X)
  rhs <- paste(c("D", sprintf("s(%s)", names(X))), collapse = " + ")
  formula <- as.formula(paste("Y ~", rhs))
  n <- nrow(dat)
  index <- lower <- upper <- numeric(n)
  alpha_obs <- numeric(n)
  alpha <- numeric(length(folds))
  for (k in seq_along(folds)) {
    test <- folds[[k]]
    train <- setdiff(seq_len(n), test)
    model <- mgcv::gam(formula, family = law$family,
                       data = dat[train, , drop = FALSE], method = method)
    eta <- as.numeric(predict(model, dat[test, , drop = FALSE], type = "link"))
    p <- law$inverse(eta)
    index[test] <- eta
    lower[test] <- ifelse(d$Y[test] == 0L, 0, 1 - p)
    upper[test] <- ifelse(d$Y[test] == 0L, 1 - p, 1)
    alpha[k] <- unname(coef(model)["D"])
    alpha_obs[test] <- alpha[k]
  }
  list(index = index, lower = lower, upper = upper,
       reference = law$reference, direct_theta = mean(alpha),
       index_control = index - alpha_obs * d$D,
       treatment_effect = alpha_obs,
       diagnostics = list(direct_theta_sd = sd(alpha)))
}

make_binary_pl_gam_adapter <- function(link = "logit", method = "REML") {
  force(link); force(method)
  function(d, folds) fit_binary_pl_gam_adapter(d, folds, link, method)
}

# Flexible proportional-odds learner. PropOdds supplies one global ordered
# threshold vector per fold through nuisance(); predictions supply category
# probabilities, which are converted directly to PIT bounds. The prediction
# contrast verifies that the treatment contribution is constant.
fit_ordinal_propodds_adapter <- function(d, folds, mstop = 2000L,
                                         nu = 0.1) {
  X <- as.data.frame(d$X)
  dat <- data.frame(Y = ordered(d$Y), D = d$D, X)
  rhs <- paste(c("bols(D)", sprintf("bbs(%s)", names(X))), collapse = " + ")
  formula <- as.formula(paste("Y ~", rhs))
  environment(formula) <- asNamespace("mboost")
  n <- nrow(dat)
  index <- lower <- upper <- numeric(n)
  alpha <- numeric(length(folds))
  all_ordered <- logical(length(folds))
  min_gap <- numeric(length(folds))
  for (k in seq_along(folds)) {
    test <- folds[[k]]
    train <- setdiff(seq_len(n), test)
    model <- mboost::gamboost(
      formula, data = dat[train, , drop = FALSE], family = mboost::PropOdds(),
      control = mboost::boost_control(mstop = mstop, nu = nu),
      baselearner = mboost::bbs
    )
    prob <- as.matrix(predict(model, newdata = dat[test, , drop = FALSE],
                              type = "response"))
    bounds <- categorical_cdf_bounds(d$Y[test], prob)
    index[test] <- as.numeric(predict(model, dat[test, , drop = FALSE],
                                      type = "link"))
    lower[test] <- bounds$lower
    upper[test] <- bounds$upper
    treated <- dat[test, , drop = FALSE]
    control <- treated
    treated$D <- 1
    control$D <- 0
    contrast <- as.numeric(predict(model, treated, type = "link") -
                             predict(model, control, type = "link"))
    alpha[k] <- mean(contrast)
    cuts <- as.numeric(mboost::nuisance(model))
    gaps <- diff(cuts)
    all_ordered[k] <- all(gaps > 0)
    min_gap[k] <- if (length(gaps)) min(gaps) else Inf
  }
  list(
    index = index, lower = lower, upper = upper, reference = "logistic",
    direct_theta = mean(alpha),
    diagnostics = list(
      direct_theta_sd = sd(alpha),
      thresholds_ordered = as.numeric(all(all_ordered)),
      min_threshold_gap = min(min_gap)
    )
  )
}

make_ordinal_propodds_adapter <- function(mstop = 2000L, nu = 0.1) {
  force(mstop); force(nu)
  function(d, folds) fit_ordinal_propodds_adapter(d, folds, mstop, nu)
}

# Parametric cumulative-link adapter with proper joint draws of coefficients
# and global thresholds. It retains the validated stage-5r behavior while
# exposing the same distribution contract as flexible families.
fit_ordinal_clm_adapter <- function(d, folds, link = "logit", df_ns = 4L) {
  X <- as.data.frame(d$X)
  basis <- do.call(cbind, lapply(X, function(x) splines::ns(x, df = df_ns)))
  colnames(basis) <- paste0("B", seq_len(ncol(basis)))
  mm <- cbind(D = d$D, basis)
  dat <- data.frame(Y = factor(d$Y), mm)
  model <- ordinal::clm(Y ~ ., data = dat, link = link)
  n_cat <- nlevels(dat$Y)
  par_hat <- c(model$alpha, model$beta)
  covariance <- vcov(model)[names(par_hat), names(par_hat), drop = FALSE]
  reference <- switch(link, logit = "logistic", probit = "normal",
                      cloglog = "gumbel_min",
                      stop("unsupported clm link: ", link))
  make_spec <- function(par) {
    cuts <- as.numeric(par[seq_len(n_cat - 1L)])
    beta <- par[n_cat:length(par)]
    index <- as.numeric(mm %*% beta)
    cdf <- switch(reference, logistic = plogis, normal = pnorm,
                  gumbel_min = pgumbel_min)
    lo_cut <- c(-Inf, cuts)[d$Y]
    hi_cut <- c(cuts, Inf)[d$Y]
    list(index = index, lower = cdf(lo_cut - index),
         upper = cdf(hi_cut - index), reference = reference,
         thresholds_ordered = all(diff(cuts) > 0))
  }
  point <- make_spec(par_hat)
  point$direct_theta <- unname(model$beta["D"])
  point$draw <- function() {
    for (attempt in seq_len(100L)) {
      par <- MASS::mvrnorm(1L, par_hat, covariance)
      spec <- make_spec(par)
      if (spec$thresholds_ordered) return(spec)
    }
    stop("failed to draw ordered cumulative-link thresholds")
  }
  point$diagnostics <- list(thresholds_ordered = 1,
                            min_threshold_gap = min(diff(model$alpha)))
  point
}

make_ordinal_clm_adapter <- function(link = "logit", df_ns = 4L) {
  force(link); force(df_ns)
  function(d, folds) fit_ordinal_clm_adapter(d, folds, link, df_ns)
}

fit_count_pl_gam_adapter <- function(d, folds,
                                     family = c("poisson", "negbin"),
                                     method = "REML",
                                     cdf_family = family) {
  family <- match.arg(family)
  cdf_family <- match.arg(cdf_family, c("poisson", "negbin"))
  X <- as.data.frame(d$X)
  dat <- data.frame(Y = as.integer(d$Y), D = d$D, X)
  stopifnot(all(dat$Y >= 0L))
  rhs <- paste(c("D", sprintf("s(%s)", names(X))), collapse = " + ")
  formula <- as.formula(paste("Y ~", rhs))
  n <- nrow(dat)
  index <- lower <- upper <- dispersion <- numeric(n)
  alpha <- numeric(length(folds))
  for (k in seq_along(folds)) {
    test <- folds[[k]]
    train <- setdiff(seq_len(n), test)
    fam <- if (family == "poisson") stats::poisson(link = "log") else
      mgcv::nb(link = "log")
    model <- mgcv::gam(formula, family = fam,
                       data = dat[train, , drop = FALSE], method = method)
    eta <- as.numeric(predict(model, dat[test, , drop = FALSE], type = "link"))
    mu <- exp(pmin(eta, 30))
    index[test] <- eta
    size <- if (family == "poisson") Inf else
      model$family$getTheta(trans = TRUE)
    if (cdf_family == "poisson") {
      lower[test] <- ppois(dat$Y[test] - 1L, lambda = mu)
      upper[test] <- ppois(dat$Y[test], lambda = mu)
      dispersion[test] <- size
    } else {
      if (!is.finite(size)) size <- .estimate_count_size(dat$Y[train],
                                                         fitted(model))
      lower[test] <- pnbinom(dat$Y[test] - 1L, mu = mu, size = size)
      upper[test] <- pnbinom(dat$Y[test], mu = mu, size = size)
      dispersion[test] <- size
    }
    alpha[k] <- unname(coef(model)["D"])
  }
  list(
    index = index, lower = lower, upper = upper, reference = "normal",
    family = family, cdf_family = cdf_family,
    direct_theta = mean(alpha),
    diagnostics = list(
      direct_theta_sd = sd(alpha),
      dispersion = if (all(is.infinite(dispersion))) Inf else
        mean(dispersion[is.finite(dispersion)]),
      min_cdf_width = min(upper - lower)
    )
  )
}

make_count_pl_gam_adapter <- function(family = c("poisson", "negbin"),
                                      method = "REML", cdf_family = family) {
  family <- match.arg(family)
  cdf_family <- match.arg(cdf_family, c("poisson", "negbin"))
  force(family); force(method); force(cdf_family)
  function(d, folds) fit_count_pl_gam_adapter(d, folds, family, method,
                                               cdf_family)
}

.estimate_count_size <- function(y, mu, lower = 0.05, upper = 1000) {
  y <- as.integer(y)
  mu <- pmax(as.numeric(mu), 1e-8)
  objective <- function(log_size) -sum(dnbinom(
    y, mu = mu, size = exp(log_size), log = TRUE
  ))
  fit <- optimize(objective, log(c(lower, upper)))
  size <- exp(fit$minimum)
  if (!is.finite(size) || size <= 0) stop("invalid count dispersion estimate")
  size
}

.count_cdf_bounds <- function(y, mu, family, size = Inf) {
  y <- as.integer(y)
  mu <- pmax(as.numeric(mu), 1e-12)
  if (family == "poisson") {
    lower <- ppois(y - 1L, lambda = mu)
    upper <- ppois(y, lambda = mu)
  } else {
    stopifnot(length(size) %in% c(1L, length(y)), all(is.finite(size)),
              all(size > 0))
    lower <- pnbinom(y - 1L, mu = mu, size = size)
    upper <- pnbinom(y, mu = mu, size = size)
  }
  list(lower = lower, upper = upper)
}

.count_fit_diagnostics <- function(y, index, lower, upper, dispersion,
                                   contrast_spread) {
  calibration <- tryCatch(coef(glm(y ~ index, family = poisson())),
                          error = function(e) c(NA_real_, NA_real_))
  list(
    dispersion = dispersion,
    min_cdf_width = min(upper - lower),
    heldout_log_score = mean(log(pmax(upper - lower, 1e-300))),
    calibration_intercept = unname(calibration[1]),
    calibration_slope = unname(calibration[2]),
    max_contrast_spread = max(contrast_spread)
  )
}

# Coefficientless Poisson-deviance learner. With interaction constraints,
# every tree uses either D alone or X alone, so the fitted scalar index is
# additive across the treatment and covariate blocks without parameterizing
# the treatment contribution as a coefficient.
fit_count_xgboost_adapter <- function(
    d, folds, family = c("poisson", "negbin"), cdf_family = family,
    constrained = TRUE, max_depth = 2L, min_child_weight = 20,
    nrounds = 500L, eta = 0.05, subsample = 0.8, nthread = 1L) {
  family <- match.arg(family)
  cdf_family <- match.arg(cdf_family, c("poisson", "negbin"))
  X <- as.data.frame(d$X)
  feature <- as.matrix(data.frame(D = d$D, X, check.names = FALSE))
  storage.mode(feature) <- "double"
  n <- nrow(feature)
  index <- lower <- upper <- dispersion_obs <- numeric(n)
  contrast_spread <- numeric(length(folds))
  constraints <- sprintf("[[0],[%s]]", paste(seq_len(ncol(X)), collapse = ","))
  for (k in seq_along(folds)) {
    test <- folds[[k]]
    train <- setdiff(seq_len(n), test)
    dtrain <- xgboost::xgb.DMatrix(feature[train, , drop = FALSE],
                                  label = d$Y[train])
    params <- list(
      objective = "count:poisson", eval_metric = "poisson-nloglik",
      max_depth = as.integer(max_depth),
      min_child_weight = min_child_weight, eta = eta,
      subsample = subsample, nthread = as.integer(nthread)
    )
    if (constrained) params$interaction_constraints <- constraints
    model <- xgboost::xgb.train(params, dtrain, nrounds = as.integer(nrounds),
                                verbose = 0)
    predict_margin <- function(matrix_value) as.numeric(predict(
      model, xgboost::xgb.DMatrix(matrix_value), outputmargin = TRUE
    ))
    eta_test <- predict_margin(feature[test, , drop = FALSE])
    mu_test <- exp(pmin(eta_test, 30))
    size <- if (cdf_family == "poisson") Inf else {
      eta_train <- predict_margin(feature[train, , drop = FALSE])
      .estimate_count_size(d$Y[train], exp(pmin(eta_train, 30)))
    }
    bounds <- .count_cdf_bounds(d$Y[test], mu_test, cdf_family, size)
    index[test] <- eta_test
    lower[test] <- bounds$lower
    upper[test] <- bounds$upper
    dispersion_obs[test] <- size
    x0 <- feature[test, , drop = FALSE]
    x1 <- x0
    x0[, 1] <- 0
    x1[, 1] <- 1
    contrast_spread[k] <- diff(range(predict_margin(x1) - predict_margin(x0)))
  }
  dispersion <- if (all(is.infinite(dispersion_obs))) Inf else
    mean(dispersion_obs[is.finite(dispersion_obs)])
  list(
    index = index, lower = lower, upper = upper, reference = "normal",
    family = family, cdf_family = cdf_family, direct_theta = NULL,
    diagnostics = .count_fit_diagnostics(
      d$Y, index, lower, upper, dispersion, contrast_spread
    )
  )
}

make_count_xgboost_adapter <- function(
    family = c("poisson", "negbin"), cdf_family = family,
    constrained = TRUE, max_depth = 2L, min_child_weight = 20,
    nrounds = 500L, eta = 0.05, subsample = 0.8, nthread = 1L) {
  family <- match.arg(family)
  cdf_family <- match.arg(cdf_family, c("poisson", "negbin"))
  force(family); force(cdf_family); force(constrained); force(max_depth)
  force(min_child_weight); force(nrounds); force(eta); force(subsample)
  force(nthread)
  function(d, folds) fit_count_xgboost_adapter(
    d, folds, family, cdf_family, constrained, max_depth,
    min_child_weight, nrounds, eta, subsample, nthread
  )
}

# Unrestricted ranger sensitivity. Regression predictions are interpreted as
# conditional means and paired with a fold-specific count CDF.
fit_count_ranger_adapter <- function(
    d, folds, family = c("poisson", "negbin"), cdf_family = family,
    num.trees = 500L, min.node.size = 20L, mtry = c("sqrt", "all")) {
  family <- match.arg(family)
  cdf_family <- match.arg(cdf_family, c("poisson", "negbin"))
  mtry <- match.arg(mtry)
  X <- as.data.frame(d$X)
  feature <- data.frame(D = d$D, X, check.names = FALSE)
  n <- nrow(feature)
  index <- lower <- upper <- dispersion_obs <- numeric(n)
  contrast_spread <- numeric(length(folds))
  mtry_value <- if (mtry == "all") ncol(feature) else
    max(1L, floor(sqrt(ncol(feature))))
  for (k in seq_along(folds)) {
    test <- folds[[k]]
    train <- setdiff(seq_len(n), test)
    train_data <- data.frame(Y = d$Y[train], feature[train, , drop = FALSE])
    model <- ranger::ranger(
      Y ~ ., data = train_data, num.trees = as.integer(num.trees),
      min.node.size = as.integer(min.node.size), mtry = mtry_value
    )
    predict_mu <- function(value) pmax(as.numeric(predict(
      model, data = data.frame(value, check.names = FALSE)
    )$predictions), 1e-8)
    mu_test <- predict_mu(feature[test, , drop = FALSE])
    size <- if (cdf_family == "poisson") Inf else
      .estimate_count_size(d$Y[train], predict_mu(feature[train, , drop = FALSE]))
    bounds <- .count_cdf_bounds(d$Y[test], mu_test, cdf_family, size)
    index[test] <- log(mu_test)
    lower[test] <- bounds$lower
    upper[test] <- bounds$upper
    dispersion_obs[test] <- size
    x0 <- feature[test, , drop = FALSE]
    x1 <- x0
    x0$D <- 0
    x1$D <- 1
    contrast_spread[k] <- diff(range(log(predict_mu(x1)) - log(predict_mu(x0))))
  }
  dispersion <- if (all(is.infinite(dispersion_obs))) Inf else
    mean(dispersion_obs[is.finite(dispersion_obs)])
  list(
    index = index, lower = lower, upper = upper, reference = "normal",
    family = family, cdf_family = cdf_family, direct_theta = NULL,
    diagnostics = .count_fit_diagnostics(
      d$Y, index, lower, upper, dispersion, contrast_spread
    )
  )
}

make_count_ranger_adapter <- function(
    family = c("poisson", "negbin"), cdf_family = family,
    num.trees = 500L, min.node.size = 20L, mtry = "sqrt") {
  family <- match.arg(family)
  cdf_family <- match.arg(cdf_family, c("poisson", "negbin"))
  force(family); force(cdf_family); force(num.trees); force(min.node.size)
  force(mtry)
  function(d, folds) fit_count_ranger_adapter(
    d, folds, family, cdf_family, num.trees, min.node.size, mtry
  )
}

make_count_oracle_adapter <- function(
    family = c("poisson", "negbin"), theta, size = 2,
    cdf_family = family, index_function = NULL) {
  family <- match.arg(family)
  cdf_family <- match.arg(cdf_family, c("poisson", "negbin"))
  force(family); force(theta); force(size); force(cdf_family)
  if (is.null(index_function)) {
    index_function <- function(d) {
      X <- as.data.frame(d$X)
      0.5 + theta * d$D + 0.35 * (X[[1]]^2 - 1) +
        0.5 * sin(X[[2]]) + 0.25 * X[[3]]
    }
  }
  force(index_function)
  function(d, folds) {
    index <- as.numeric(index_function(d))
    cdf_size <- if (cdf_family == "poisson") Inf else size
    bounds <- .count_cdf_bounds(d$Y, exp(pmin(index, 30)), cdf_family,
                                cdf_size)
    list(
      index = index, lower = bounds$lower, upper = bounds$upper,
      reference = "normal", family = family, cdf_family = cdf_family,
      direct_theta = theta,
      diagnostics = .count_fit_diagnostics(
        d$Y, index, bounds$lower, bounds$upper, cdf_size, 0
      )
    )
  }
}

.component_projection <- function(value, X, D_res, fit_l, folds) {
  nuisance <- crossfit(X, value, fit_l, folds)
  fit <- fwl_theta(value - nuisance, D_res)
  c(fit, list(nuisance = nuisance, residual = value - nuisance))
}

# Lightweight Rao-Blackwell target used by target-level tuning and bias-first
# ablations. It deliberately omits randomized draws and comparators.
count_rb_point <- function(d, adapter, folds, fit_l = fit_gam, fit_m = NULL) {
  X <- as.data.frame(d$X)
  if (is.null(fit_m)) {
    fit_m <- if (all(d$D %in% c(0, 1))) fit_gam_binomial else fit_gam
  }
  full <- adapter(d, folds)
  .check_discrete_fit(full, nrow(X))
  D_res <- d$D - crossfit(X, d$D, fit_m, folds)
  index_fit <- .component_projection(full$index, X, D_res, fit_l, folds)
  correction_fit <- .component_projection(
    normal_jump_mean(full$lower, full$upper), X, D_res, fit_l, folds
  )
  list(
    theta = index_fit$theta + correction_fit$theta,
    index_only = index_fit$theta, correction = correction_fit$theta,
    full = full, D_res = D_res
  )
}

# Tailored Poisson orthogonal one-step comparator. It uses the weighted
# residual D-a(X), where a(X)=E[D exp(theta D)|X]/E[exp(theta D)|X], and a
# cross-fitted offset Poisson regression for g(X).
count_poisson_one_step <- function(d, folds, theta_initial, fit_l = fit_gam) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  mu <- a_hat <- numeric(n)
  rhs <- paste(sprintf("s(%s)", names(X)), collapse = " + ")
  for (test in folds) {
    train <- setdiff(seq_len(n), test)
    dat_train <- data.frame(Y = d$Y[train], D = d$D[train],
                            X[train, , drop = FALSE])
    dat_train$offset_value <- theta_initial * dat_train$D
    model <- mgcv::gam(
      as.formula(paste("Y ~ offset(offset_value) +", rhs)),
      family = poisson(link = "log"), data = dat_train, method = "REML"
    )
    dat_test <- data.frame(D = d$D[test], X[test, , drop = FALSE])
    dat_test$offset_value <- theta_initial * dat_test$D
    mu[test] <- as.numeric(predict(model, dat_test, type = "response"))
    tilt <- exp(pmin(theta_initial * d$D[train], 30))
    denominator_fit <- fit_l(X[train, , drop = FALSE], tilt)
    numerator_fit <- fit_l(X[train, , drop = FALSE], d$D[train] * tilt)
    denominator <- pmax(denominator_fit(X[test, , drop = FALSE]), 1e-8)
    a_hat[test] <- numerator_fit(X[test, , drop = FALSE]) / denominator
  }
  weight <- d$D - a_hat
  score_i <- weight * (d$Y - mu)
  bread <- sum(weight * d$D * mu)
  if (!is.finite(bread) || abs(bread) < 1e-10) {
    return(list(theta = NA_real_, se = NA_real_))
  }
  theta <- theta_initial + sum(score_i) / bread
  se <- sqrt(sum(score_i^2)) / abs(bread)
  list(theta = theta, se = se)
}

sudo_ordinal <- function(d, adapter = make_ordinal_clm_adapter(), B = 25L,
                         n_folds = 5L, folds = NULL, ...) {
  sudo_discrete(d, adapter, B, n_folds, folds,
                refit_S_nuisance = TRUE, ...)
}

sudo_count <- function(
    d, adapter = make_count_pl_gam_adapter("poisson"), B = 25L,
    n_folds = 5L, folds = NULL, fit_l = fit_gam, fit_m = NULL,
    outcome_family = c("poisson", "negbin"), level = 0.95,
    structured_comparator = TRUE, one_step = TRUE,
    refit_randomized_nuisance = FALSE, include_index_only = TRUE,
    include_raw_label = TRUE, ...) {
  outcome_family <- match.arg(outcome_family)
  stopifnot(is.function(adapter), B >= 2L, n_folds >= 2L)
  X <- as.data.frame(d$X)
  n <- nrow(X)
  if (is.null(folds)) folds <- make_folds(n, n_folds)
  stopifnot(identical(sort(as.integer(unlist(folds))), seq_len(n)))
  if (is.null(fit_m)) {
    fit_m <- if (all(d$D %in% c(0, 1))) fit_gam_binomial else fit_gam
  }
  full <- adapter(d, folds)
  .check_discrete_fit(full, n)
  D_res <- d$D - crossfit(X, d$D, fit_m, folds)
  jump_mean <- normal_jump_mean(full$lower, full$upper)
  if (include_index_only) {
    index_fit <- .component_projection(full$index, X, D_res, fit_l, folds)
    correction_fit <- .component_projection(jump_mean, X, D_res, fit_l, folds)
    rb_nuisance <- index_fit$nuisance + correction_fit$nuisance
    rb_residual <- index_fit$residual + correction_fit$residual
    rb_fit <- fwl_theta(rb_residual, D_res)
    stopifnot(abs(rb_fit$theta - index_fit$theta - correction_fit$theta) < 1e-10)
  } else {
    rb_value <- full$index + jump_mean
    rb_fit <- .component_projection(rb_value, X, D_res, fit_l, folds)
    rb_nuisance <- rb_fit$nuisance
    index_fit <- list(theta = NA_real_, se = NA_real_)
    correction_fit <- list(theta = NA_real_)
  }

  draws <- vapply(seq_len(B), function(b) {
    surrogate <- complete_discrete(full$lower, full$upper, full$index,
                                   "normal")
    # Reusing the analytic nuisance makes the randomized arm a finite-B
    # approximation to the Rao-Blackwell target. Set the flag to TRUE only
    # for a per-completion nuisance-refit sensitivity analysis.
    if (refit_randomized_nuisance) {
      projected <- .component_projection(surrogate, X, D_res, fit_l, folds)
      c(theta = projected$theta, var = projected$var)
    } else {
      projected <- fwl_theta(surrogate - rb_nuisance, D_res)
      c(theta = projected$theta, var = projected$var)
    }
  }, numeric(2))
  randomized <- pool_rubin(draws["theta", ], draws["var", ], level = level,
                           n_obs = n)

  gam_fit <- if (structured_comparator)
    fit_count_pl_gam_adapter(d, folds, outcome_family) else NULL
  direct_theta <- if (is.null(gam_fit)) NA_real_ else gam_fit$direct_theta
  poisson_os <- if (one_step) count_poisson_one_step(
    d, folds,
    theta_initial = if (is.finite(direct_theta)) direct_theta else rb_fit$theta,
    fit_l = fit_l
  ) else list(theta = NA_real_, se = NA_real_)
  raw <- if (include_raw_label) plr_crossfit(
    d$Y, d$D, X, fit_l = fit_l, fit_m = fit_m, folds = folds
  ) else list(theta = NA_real_)
  targets <- c(
    rao_blackwell = rb_fit$theta,
    randomized = randomized$theta
  )
  if (include_index_only) targets <- c(targets, index_only = index_fit$theta)
  if (one_step) targets <- c(targets, poisson_one_step = poisson_os$theta)
  target_internal_se <- c(
    rao_blackwell = rb_fit$se,
    randomized = randomized$se
  )
  if (include_index_only) target_internal_se <- c(
    target_internal_se, index_only = index_fit$se
  )
  if (one_step) target_internal_se <- c(
    target_internal_se, poisson_one_step = poisson_os$se
  )
  out <- list(
    theta = rb_fit$theta, se = rb_fit$se,
    ci_lo = rb_fit$theta - qnorm(1 - (1 - level) / 2) * rb_fit$se,
    ci_hi = rb_fit$theta + qnorm(1 - (1 - level) / 2) * rb_fit$se,
    targets = targets, target_internal_se = target_internal_se,
    rao_blackwell_theta = rb_fit$theta,
    randomized_theta = randomized$theta,
    randomized_se = randomized$se,
    index_only_theta = index_fit$theta,
    outcome_correction = correction_fit$theta,
    poisson_one_step_theta = poisson_os$theta,
    structured_gam_theta = direct_theta,
    direct_theta = direct_theta,
    raw_label_theta = raw$theta,
    completion_difference = randomized$theta - rb_fit$theta,
    max_abs_index = max(abs(full$index))
  )
  if (!is.null(full$diagnostics)) out <- c(out, full$diagnostics)
  out
}
