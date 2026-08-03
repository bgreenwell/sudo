# Bikeshare application: an illustrative conditional count association.
#
# Outcome: total hourly rentals. Treatment: temperature standardized over the
# sample. The target exp(theta) is a conditional rental-rate ratio per one-SD
# temperature increase. atemp, casual, and registered are excluded. The
# primary imputation learner is coefficientless constrained XGBoost with a
# negative-binomial CDF. Whole dates stay together in five contiguous folds.
# Inference uses a moving-block full-pipeline bootstrap with 7-day blocks and
# 14/28-day sensitivity.
#
# The block bootstrap is empirical here because current SUDO theory assumes
# independent observations. This application does not claim a settled causal
# temperature effect.

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/discrete.R")
source("R/sudo/estimator.R")
source("R/sudo/count_validation.R")

env_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}
B_outer <- env_int("SUDO_BIKE_OUTER", 199L)
B <- env_int("SUDO_BIKE_B", 25L)
n_folds <- env_int("SUDO_BIKE_FOLDS", 5L)
max_days <- env_int("SUDO_BIKE_MAX_DAYS", 0L)
xgb_rounds_smoke <- env_int("SUDO_BIKE_XGB_ROUNDS", 30L)
block_value <- Sys.getenv("SUDO_BIKE_BLOCKS", unset = "")
block_grid <- if (nzchar(block_value))
  as.integer(strsplit(block_value, ",")[[1]]) else c(7L, 14L, 28L)
data_dir <- "manuscript/data/bikeshare"
data_file <- file.path(data_dir, "hour.csv")
if (!file.exists(data_file)) {
  dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
  archive <- tempfile(fileext = ".zip")
  download.file(
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    archive, mode = "wb", quiet = TRUE
  )
  unzip(archive, files = "hour.csv", exdir = data_dir)
}

raw <- read.csv(data_file, stringsAsFactors = FALSE)
stopifnot(all(c("cnt", "temp", "dteday", "hr", "yr", "weekday",
                "holiday", "workingday", "weathersit", "hum",
                "windspeed") %in% names(raw)))
raw$date <- as.Date(raw$dteday)
if (max_days > 0L) {
  keep_dates <- head(sort(unique(raw$date)), max_days)
  raw <- raw[raw$date %in% keep_dates, , drop = FALSE]
}
raw$doy <- as.integer(format(raw$date, "%j"))
raw$date_id <- as.integer(factor(raw$date, levels = sort(unique(raw$date))))
raw$D <- as.numeric(scale(raw$temp))

# X retains the raw terms needed by the structured count adapter. The
# nuisance GAMs receive a stable numeric encoding with cyclic harmonics.
X <- data.frame(
  date_id = raw$date_id, hour = raw$hr, doy = raw$doy, year = raw$yr,
  weekday = raw$weekday, holiday = raw$holiday,
  workingday = raw$workingday, weather = raw$weathersit,
  humidity = raw$hum, windspeed = raw$windspeed,
  hour_sin = sin(2 * pi * raw$hr / 24),
  hour_cos = cos(2 * pi * raw$hr / 24),
  doy_sin = sin(2 * pi * raw$doy / 366),
  doy_cos = cos(2 * pi * raw$doy / 366)
)
d <- list(X = X, D = raw$D, Y = raw$cnt)

full_tuning_file <- "R/results/stage15t_count_selected.csv"
full_count_file <- "R/results/stage15_count.csv"
full_oracle_file <- "R/results/stage15o_count_oracle.csv"
is_full_application <- B_outer >= 199L && B >= 25L &&
  length(unique(d$X$date_id)) >= 700L
if (is_full_application &&
    (!file.exists(full_tuning_file) || !file.exists(full_count_file) ||
       !file.exists(full_oracle_file))) {
  stop("full bikeshare analysis requires passed full stage-15 oracle, ",
       "tuning, and confirmation result files")
}
if (file.exists(full_tuning_file)) {
  selected <- read.csv(full_tuning_file, stringsAsFactors = FALSE)
  xrow <- selected[selected$learner == "xgboost", , drop = FALSE]
  stopifnot(nrow(xrow) == 1L)
  xgb_config <- list(max_depth = xrow$max_depth,
                     min_child_weight = xrow$min_child_weight,
                     nrounds = xrow$nrounds)
} else {
  xgb_config <- list(max_depth = 2L, min_child_weight = 20,
                     nrounds = xgb_rounds_smoke)
}
if (is_full_application) {
  confirmation <- read.csv(full_count_file, stringsAsFactors = FALSE)
  oracle_confirmation <- read.csv(full_oracle_file, stringsAsFactors = FALSE)
  primary <- confirmation[
    confirmation$arm == "xgb_constrained" &
      confirmation$estimator %in% c("rao_blackwell", "randomized"),
    , drop = FALSE
  ]
  stopifnot(
    nrow(primary) == 12L, all(count_primary_gates(primary)$pass),
    nrow(oracle_confirmation) == 12L,
    all(count_primary_gates(oracle_confirmation)$pass),
    all(abs(oracle_confirmation$completion_mean_difference) <=
          2 * oracle_confirmation$completion_mc_se)
  )
}

contiguous_date_folds <- function(d, k) {
  dates <- sort(unique(d$X$date_id))
  groups <- split(dates, cut(seq_along(dates), breaks = k, labels = FALSE))
  lapply(groups, function(value) which(d$X$date_id %in% value))
}

fit_bikeshare_adapter <- function(d, folds) {
  x <- as.data.frame(d$X)
  dat <- data.frame(
    Y = d$Y, D = d$D, hour = x$hour, doy = x$doy,
    year = x$year, weekday = factor(x$weekday),
    holiday = x$holiday, workingday = x$workingday,
    weather = pmin(x$weather, 3), humidity = x$humidity,
    windspeed = x$windspeed
  )
  formula <- Y ~ D + s(hour, bs = "cc", k = 12) +
    s(doy, bs = "cc", k = 10) + year + weekday + holiday + workingday +
    weather + s(humidity, k = 5) + s(windspeed, k = 5)
  n <- nrow(dat)
  index <- lower <- upper <- size_obs <- numeric(n)
  alpha <- numeric(length(folds))
  for (fold in seq_along(folds)) {
    test <- folds[[fold]]
    train <- setdiff(seq_len(n), test)
    model <- mgcv::gam(
      formula, family = mgcv::nb(link = "log"), method = "REML",
      knots = list(hour = c(0, 24), doy = c(0.5, 366.5)),
      data = dat[train, , drop = FALSE]
    )
    eta <- as.numeric(predict(model, dat[test, , drop = FALSE], type = "link"))
    mu <- exp(pmin(eta, 30))
    size <- model$family$getTheta(trans = TRUE)
    index[test] <- eta
    lower[test] <- pnbinom(dat$Y[test] - 1L, mu = mu, size = size)
    upper[test] <- pnbinom(dat$Y[test], mu = mu, size = size)
    size_obs[test] <- size
    alpha[fold] <- unname(coef(model)["D"])
  }
  list(
    index = index, lower = lower, upper = upper, reference = "normal",
    direct_theta = mean(alpha),
    diagnostics = list(direct_theta_sd = sd(alpha),
                       dispersion = mean(size_obs),
                       min_cdf_width = min(upper - lower))
  )
}

fit_l_bike <- function(X, y) {
  keep <- c("hour_sin", "hour_cos", "doy_sin", "doy_cos", "year",
            "weekday", "holiday", "workingday", "weather", "humidity",
            "windspeed")
  dat <- data.frame(y = y, X[, keep, drop = FALSE])
  model <- mgcv::gam(
    y ~ hour_sin + hour_cos + doy_sin + doy_cos + year + weekday +
      holiday + workingday + weather + s(humidity, k = 5) +
      s(windspeed, k = 5),
    data = dat, method = "REML"
  )
  function(Xnew) as.numeric(predict(
    model, newdata = data.frame(Xnew[, keep, drop = FALSE])
  ))
}
fit_m_bike <- fit_l_bike

moving_block_indices <- function(block_days) {
  force(block_days)
  function(d, bootstrap_id) {
    dates <- sort(unique(d$X$date_id))
    n_dates <- length(dates)
    starts <- sample.int(n_dates, ceiling(n_dates / block_days), replace = TRUE)
    sampled_dates <- unlist(lapply(starts, function(start) {
      dates[((start - 1L + seq_len(block_days) - 1L) %% n_dates) + 1L]
    }), use.names = FALSE)[seq_len(n_dates)]
    unlist(lapply(sampled_dates, function(day) which(d$X$date_id == day)),
           use.names = FALSE)
  }
}

fit_application <- function(block_days) {
  adapter <- make_count_xgboost_adapter(
    "negbin", constrained = TRUE,
    max_depth = xgb_config$max_depth,
    min_child_weight = xgb_config$min_child_weight,
    nrounds = xgb_config$nrounds
  )
  sudo_pipeline_boot(
    d, B_outer = B_outer, inner_B = B, n_folds = n_folds,
    estimator = sudo_count, adapter = adapter,
    fit_l = fit_l_bike, fit_m = fit_m_bike,
    outcome_family = "negbin", structured_comparator = FALSE,
    one_step = FALSE,
    fold_builder = contiguous_date_folds,
    resample_indices = moving_block_indices(block_days)
  )
}

# Held-out randomized-quantile diagnostics use an RNG stream independent of
# estimation and bootstrap draws.
set.seed(275001)
folds <- contiguous_date_folds(d, n_folds)
diagnostic_fit <- make_count_xgboost_adapter(
  "negbin", constrained = TRUE,
  max_depth = xgb_config$max_depth,
  min_child_weight = xgb_config$min_child_weight,
  nrounds = xgb_config$nrounds
)(d, folds)
structured_fit <- fit_bikeshare_adapter(d, folds)
rqr <- complete_discrete(diagnostic_fit$lower, diagnostic_fit$upper,
                         rep(0, length(d$Y)), "normal")
diagnostics <- data.frame(
  n_obs = length(d$Y), n_dates = length(unique(d$X$date_id)),
  rqr_mean = mean(rqr), rqr_sd = sd(rqr),
  rqr_index_correlation = cor(rqr, diagnostic_fit$index),
  zero_rate = mean(d$Y == 0L), max_count = max(d$Y),
  heldout_log_score = diagnostic_fit$diagnostics$heldout_log_score,
  dispersion = diagnostic_fit$diagnostics$dispersion,
  calibration_intercept = diagnostic_fit$diagnostics$calibration_intercept,
  calibration_slope = diagnostic_fit$diagnostics$calibration_slope,
  constrained_max_contrast_spread =
    diagnostic_fit$diagnostics$max_contrast_spread
)

set.seed(275100)
results <- do.call(rbind, lapply(block_grid, function(block_days) {
  fit <- fit_application(block_days)
  target <- fit$target_results
  get <- function(name, column) target[[column]][target$target == name]
  data.frame(
    n_obs = length(d$Y), n_dates = length(unique(d$X$date_id)),
    B_outer = B_outer, B = B, block_days = block_days,
    rao_blackwell_theta = get("rao_blackwell", "estimate"),
    rao_blackwell_se = get("rao_blackwell", "se"),
    rao_blackwell_ci_lo = get("rao_blackwell", "ci_lo"),
    rao_blackwell_ci_hi = get("rao_blackwell", "ci_hi"),
    randomized_theta = get("randomized", "estimate"),
    randomized_se = get("randomized", "se"),
    randomized_ci_lo = get("randomized", "ci_lo"),
    randomized_ci_hi = get("randomized", "ci_hi"),
    index_only_theta = get("index_only", "estimate"),
    index_only_se = get("index_only", "se"),
    rao_blackwell_rate_ratio = exp(get("rao_blackwell", "estimate")),
    randomized_rate_ratio = exp(get("randomized", "estimate")),
    direct_negbin_gam_theta = structured_fit$direct_theta,
    direct_negbin_gam_rate_ratio = exp(structured_fit$direct_theta),
    dispersion = fit$dispersion
  )
}))
print(results, row.names = FALSE)
print(diagnostics, row.names = FALSE)
dir.create("R/results", showWarnings = FALSE)
full_run <- is_full_application
suffix <- if (full_run) "" else "_smoke"
write.csv(results, paste0("R/results/bikeshare_application", suffix, ".csv"),
          row.names = FALSE)
write.csv(diagnostics, paste0("R/results/bikeshare_diagnostics", suffix, ".csv"),
          row.names = FALSE)
cat("\nInterpretation: illustrative conditional association only. The moving-",
    "block bootstrap is empirical because current theory assumes independent",
    "observations.\n")
