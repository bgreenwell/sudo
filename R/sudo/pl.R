# Partially-linear (PL) full models: V = alpha*D + f(X), D structurally
# forbidden from interacting with X, six different ways. Each fitter
# returns the same list(lp_hat, fit_fold, to_lp) shape fit_blackbox_index()
# does, so sudo_binary()'s dispatch, nuisance fitting, recentered-bootstrap
# draws, and Rubin pooling work unchanged for every variant.
# Optional variants require earth, ranger, nnet, xgboost, or mboost;
# fwl.R/estimator.R must be sourced first.

# ---- (0) native GAM: parametric D plus smooths of X -----------------------
#
# This is the same structural model used by crossfit_fullmodel_gam(), exposed
# through the black-box fitter contract so a full-pipeline bootstrap can refit
# it without also injecting coefficient draws inside each resample.
fit_pl_gam <- function(d, folds, method = "GCV.Cp",
                       link = c("logit", "probit", "cloglog")) {
  link <- match.arg(link)
  X <- as.data.frame(d$X)
  n <- nrow(X)
  dat <- data.frame(Y = d$Y, D = d$D, X)
  rhs <- paste(c("D", sprintf("s(%s)", colnames(X))), collapse = " + ")
  fml <- as.formula(paste("Y ~", rhs))

  fit_fold <- function(train_idx, test) {
    m <- mgcv::gam(fml, family = binomial(link = link),
                   data = dat[train_idx, , drop = FALSE],
                   method = method)
    as.numeric(predict(m, newdata = dat[test, , drop = FALSE],
                       type = "response"))
  }

  to_lp <- switch(
    link,
    logit = function(p) qlogis(pmin(pmax(p, 1e-5), 1 - 1e-5)),
    probit = function(p) qnorm(pmin(pmax(p, 1e-5), 1 - 1e-5)),
    cloglog = function(p) log(-log1p(-pmin(pmax(p, 1e-5), 1 - 1e-5)))
  )
  p_hat <- numeric(n)
  for (test in folds) p_hat[test] <- fit_fold(setdiff(seq_len(n), test), test)
  list(lp_hat = to_lp(p_hat), fit_fold = fit_fold, to_lp = to_lp)
}

# ---- (1) deterministic additive series: mandatory D + Hermite bases ------
#
# This is the theorem-reference learner. The basis is fixed before seeing Y:
# normalized probabilists' Hermite polynomials of degrees 1,...,degree are
# applied separately to each X column. The caller is responsible for placing
# X on the scale for which this dictionary is intended. In the validation
# design X is standard normal, so the columns are orthonormal in population.
# D is added only after the X dictionary is built and therefore cannot enter
# a hinge, polynomial, or treatment-by-covariate interaction.
hermite_series_basis <- function(X, degree = 3L) {
  X <- as.matrix(X)
  degree <- as.integer(degree)
  stopifnot(degree >= 1L, all(is.finite(X)))
  n <- nrow(X)
  p <- ncol(X)
  xnames <- colnames(X)
  if (is.null(xnames)) xnames <- paste0("X", seq_len(p))
  out <- matrix(
    0, nrow = n, ncol = p * degree,
    dimnames = list(
      NULL,
      unlist(lapply(xnames, function(x) paste0(x, "_h", seq_len(degree))))
    )
  )
  for (j in seq_len(p)) {
    x <- X[, j]
    h_prev <- rep(1, n)
    h_cur <- x
    out[, (j - 1L) * degree + 1L] <- h_cur
    if (degree >= 2L) {
      for (r in 2:degree) {
        h_next <- (x * h_cur - sqrt(r - 1) * h_prev) / sqrt(r)
        out[, (j - 1L) * degree + r] <- h_next
        h_prev <- h_cur
        h_cur <- h_next
      }
    }
  }
  out
}

fit_pl_series <- function(d, folds, degree = 3L) {
  basis <- hermite_series_basis(d$X, degree = degree)
  n <- nrow(basis)
  design <- cbind(`(Intercept)` = 1, D = d$D, basis)

  fit_fold <- function(train_idx, test) {
    g <- glm.fit(
      x = design[train_idx, , drop = FALSE],
      y = d$Y[train_idx],
      family = binomial()
    )
    coefficients <- g$coefficients
    coefficients[is.na(coefficients)] <- 0
    eta <- as.numeric(design[test, , drop = FALSE] %*% coefficients)
    plogis(eta)
  }

  to_lp <- function(p) qlogis(pmin(pmax(p, 1e-5), 1 - 1e-5))
  p_hat <- numeric(n)
  for (test in folds) {
    p_hat[test] <- fit_fold(setdiff(seq_len(n), test), test)
  }
  list(
    lp_hat = to_lp(p_hat),
    fit_fold = fit_fold,
    to_lp = to_lp,
    treatment_basis_degree = 1L,
    series_degree = as.integer(degree)
  )
}

# ---- (2) backfitting GPLM: IRLS + black-box smoother for f(X) ------------
#
# Standard binomial-GPLM backfitting (Hastie & Tibshirani 1990 style):
# at each iteration form the IRLS working response/weights at the current
# fit, regress (working response - beta*D) on X with the black-box
# learner (weighted) to update f, then weighted-LS update beta from
# (working response - f). engine = "nnet" (default: the untuned/flexible
# decay=0.01 config that produced stage 3f's worst bias, the sharpest
# test of whether PL structure alone rescues it) or "ranger".
fit_pl_backfit <- function(d, folds, engine = c("nnet", "ranger"),
                           nn_size = 8, nn_decay = 0.01, n_iter = 5,
                           link = c("logit", "probit")) {
  engine <- match.arg(engine)
  # the index must be on the same scale as the surrogate's error law, so the
  # IRLS link has to match complete_surrogate()'s. A logit backfit scored
  # against a probit completion is out by the usual ~1.8 factor.
  link <- match.arg(link)
  linkinv <- if (link == "logit") plogis else pnorm
  linkfun <- if (link == "logit") qlogis else qnorm
  X <- as.data.frame(d$X)
  n <- nrow(X)

  fit_f <- function(Xtr, target, w, Xte) {
    if (engine == "nnet") {
      m <- nnet::nnet(x = Xtr, y = target, weights = w, size = nn_size,
                      decay = nn_decay, maxit = 300, linout = TRUE,
                      trace = FALSE)
      list(tr = as.numeric(predict(m, Xtr)), te = as.numeric(predict(m, Xte)))
    } else {
      dtr <- data.frame(y = target, Xtr)
      m <- ranger::ranger(y ~ ., data = dtr, case.weights = w,
                          num.trees = 200, min.node.size = 5)
      list(tr = predict(m, data = Xtr)$predictions,
           te = predict(m, data = Xte)$predictions)
    }
  }

  backfit_fold <- function(train_idx, test) {
    Xtr <- X[train_idx, , drop = FALSE]; Dtr <- d$D[train_idx]
    Ytr <- d$Y[train_idx]; Xte <- X[test, , drop = FALSE]; Dte <- d$D[test]
    beta <- 0
    f_tr <- rep(linkfun(pmin(pmax(mean(Ytr), 1e-3), 1 - 1e-3)),
               length(train_idx))
    f_te <- rep(f_tr[1], length(test))
    for (it in seq_len(n_iter)) {
      eta <- beta * Dtr + f_tr
      mu <- pmin(pmax(linkinv(eta), 1e-4), 1 - 1e-4)
      # general IRLS: w = (dmu/deta)^2 / V(mu), z = eta + (y - mu)/(dmu/deta),
      # which collapses to the familiar mu(1-mu) form under a logit
      dm <- if (link == "logit") mu * (1 - mu) else pmax(dnorm(eta), 1e-8)
      w <- dm^2 / (mu * (1 - mu))
      z <- eta + (Ytr - mu) / dm
      out <- fit_f(Xtr, z - beta * Dtr, w, Xte)
      f_tr <- out$tr; f_te <- out$te
      beta <- unname(coef(lm(I(z - f_tr) ~ Dtr, weights = w))["Dtr"])
    }
    linkinv(beta * Dte + f_te)
  }

  to_lp <- function(p) linkfun(pmin(pmax(p, 1e-5), 1 - 1e-5))
  p_hat <- numeric(n)
  for (test in folds) p_hat[test] <- backfit_fold(setdiff(seq_len(n), test),
                                                  test)
  list(lp_hat = to_lp(p_hat), fit_fold = backfit_fold, to_lp = to_lp)
}

# ---- (3) XGBoost with interaction_constraints -----------------------------
#
# D is placed in its own singleton interaction group, so no tree path can
# combine a split on D with a split on any X feature: the ensemble
# decomposes exactly into f(X) + h(D), and h(D) collapses to one additive
# offset since D is binary. One fit, no iteration; deliberately not
# heavily tuned (moderate depth/rounds) since the point is testing whether
# the structural constraint alone protects an under-tuned model.
fit_pl_xgboost <- function(d, folds, max_depth = 4, eta = 0.1,
                           nrounds = 100) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  p <- ncol(X)
  feat <- as.matrix(cbind(X, D = d$D))
  ic <- list(0:(p - 1), p)  # X columns grouped; D alone (0-based indices)

  fit_fold <- function(train_idx, test) {
    dtr <- xgboost::xgb.DMatrix(feat[train_idx, , drop = FALSE],
                                label = d$Y[train_idx])
    bst <- xgboost::xgb.train(
      params = list(objective = "binary:logistic", max_depth = max_depth,
                    eta = eta, interaction_constraints = ic),
      data = dtr, nrounds = nrounds, verbose = 0)
    predict(bst, feat[test, , drop = FALSE])
  }

  to_lp <- function(p) qlogis(pmin(pmax(p, 1e-5), 1 - 1e-5))
  p_hat <- numeric(n)
  for (test in folds) p_hat[test] <- fit_fold(setdiff(seq_len(n), test),
                                              test)
  list(lp_hat = to_lp(p_hat), fit_fold = fit_fold, to_lp = to_lp,
       .last_bst_feat_names = colnames(feat))
}

# ---- (4) mboost componentwise gradient boosting ---------------------------
#
# gamboost(Y ~ bols(D) + bbs(X1) + ... + bbs(X5)): D gets a plain linear
# (ordinary-least-squares) base-learner, each X_j its own P-spline
# base-learner. No interaction base-learner exists in the formula, so D
# structurally cannot pick up interaction effects regardless of mstop.
# mstop chosen via cvrisk() once per fold (not per bootstrap draw, see
# stage3l timing note) unless too expensive, in which case a fixed
# generous mstop is used (documented, not silent).
# Note: "beyond boundary.knots" warnings are expected and benign. A
# fold's held-out X sometimes falls outside the training spline knot
# range; mboost linearly extrapolates, which is standard practice.
fit_pl_mboost <- function(d, folds, use_cvrisk = TRUE, fixed_mstop = 200) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  xnames <- colnames(X)
  rhs <- paste(c("bols(D)", sprintf("bbs(%s)", xnames)), collapse = " + ")
  fml <- as.formula(paste("Y ~", rhs))

  fit_fold <- function(train_idx, test) {
    dat <- data.frame(Y = factor(d$Y, levels = c(0, 1)), D = d$D, X)
    m <- mboost::gamboost(fml, data = dat[train_idx, , drop = FALSE],
                          family = mboost::Binomial(),
                          control = mboost::boost_control(mstop = fixed_mstop))
    if (use_cvrisk) {
      cv <- tryCatch(
        mboost::cvrisk(m, folds = mboost::cv(model.weights(m), type = "kfold",
                                             B = 5), papply = lapply),
        error = function(e) NULL)
      if (!is.null(cv)) m <- m[mstop(cv)]
    }
    as.numeric(predict(m, newdata = dat[test, , drop = FALSE],
                       type = "response"))
  }

  to_lp <- function(p) qlogis(pmin(pmax(p, 1e-5), 1 - 1e-5))
  p_hat <- numeric(n)
  for (test in folds) p_hat[test] <- fit_fold(setdiff(seq_len(n), test),
                                              test)
  list(lp_hat = to_lp(p_hat), fit_fold = fit_fold, to_lp = to_lp)
}

# ---- (5) MARS basis for X plus a mandatory linear D coefficient -----------
#
# earth::linpreds makes a predictor linear but still permits that predictor
# inside interaction terms. That is not the restriction needed here. Instead,
# earth selects a basis from X alone, and a binomial GLM jointly estimates an
# unpenalized D coefficient and the selected X-basis coefficients. Thus D is
# present in every fold and cannot occur in a hinge or interaction basis.
fit_pl_mars <- function(d, folds, degree = 2, nk = 41, penalty = 3) {
  X <- as.data.frame(d$X)
  n <- nrow(X)

  fit_fold <- function(train_idx, test) {
    mars <- earth::earth(
      x = X[train_idx, , drop = FALSE], y = d$Y[train_idx],
      degree = degree, nk = nk, penalty = penalty,
      glm = list(family = binomial), keepxy = TRUE)
    bx_tr <- stats::model.matrix(mars, x = X[train_idx, , drop = FALSE])
    bx_te <- stats::model.matrix(mars, x = X[test, , drop = FALSE])
    # earth's first basis column is the intercept. glm.fit receives its own
    # explicit intercept followed by mandatory D and the selected X bases.
    x_tr <- cbind(`(Intercept)` = 1, D = d$D[train_idx],
                  bx_tr[, -1, drop = FALSE])
    x_te <- cbind(`(Intercept)` = 1, D = d$D[test],
                  bx_te[, -1, drop = FALSE])
    g <- glm.fit(x = x_tr, y = d$Y[train_idx], family = binomial())
    coefficients <- g$coefficients
    coefficients[is.na(coefficients)] <- 0
    eta <- as.numeric(x_te %*% coefficients)
    plogis(eta)
  }

  to_lp <- function(p) qlogis(pmin(pmax(p, 1e-5), 1 - 1e-5))
  p_hat <- numeric(n)
  for (test in folds) p_hat[test] <- fit_fold(setdiff(seq_len(n), test), test)
  list(lp_hat = to_lp(p_hat), fit_fold = fit_fold, to_lp = to_lp,
       treatment_basis_degree = 1L)
}

# unified dispatcher used by the stage 3j/3k/3l diagnostic scripts (index
# calibration, variance ratio) so they can call any PL variant uniformly
fit_pl <- function(d, folds,
                   variant = c("gam", "series", "backfit", "xgboost",
                               "mboost", "mars"),
                   pl_engine = "nnet", pl_use_cvrisk = TRUE,
                   pl_fixed_mstop = 200,
                   mars_degree = 2, mars_nk = 41, mars_penalty = 3,
                   series_degree = 3L, link = "logit") {
  variant <- match.arg(variant)
  switch(variant,
        gam = fit_pl_gam(d, folds, link = link),
        series = fit_pl_series(d, folds, degree = series_degree),
        backfit = fit_pl_backfit(d, folds, engine = pl_engine),
        xgboost = fit_pl_xgboost(d, folds),
        mboost  = fit_pl_mboost(d, folds, use_cvrisk = pl_use_cvrisk,
                                fixed_mstop = pl_fixed_mstop),
        mars = fit_pl_mars(d, folds, degree = mars_degree, nk = mars_nk,
                           penalty = mars_penalty))
}
