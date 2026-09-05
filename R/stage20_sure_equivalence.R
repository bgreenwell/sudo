# Stage 20: Equivalence of SUDO and sure latent surrogate responses.
#
# Run from the repository root:
#   Rscript R/stage20_sure_equivalence.R
#
# The comparison is conditional on a fitted model. It does not redraw fitted
# model parameters, cross-fit nuisances, or estimate a treatment effect.

stopifnot(requireNamespace("pkgload", quietly = TRUE))
stopifnot(requireNamespace("ordinal", quietly = TRUE))

sure_source <- Sys.getenv("SURE_SOURCE", unset = file.path("..", "..", "r", "sure"))
sure_source <- normalizePath(sure_source, mustWork = FALSE)
stopifnot(file.exists(file.path(sure_source, "DESCRIPTION")))
pkgload::load_all(sure_source, quiet = TRUE, export_all = FALSE)
stopifnot(utils::packageVersion("sure") >= "0.3.0.9000")
sure_version <- as.character(utils::packageVersion("sure"))

source("R/sudo/surrogate.R")

tolerance <- 1e-10
rows <- list()

record_check <- function(outcome, link, comparison, difference, expected) {
  rows[[length(rows) + 1L]] <<- data.frame(
    sure_version = sure_version,
    outcome = outcome,
    link = link,
    comparison = comparison,
    max_abs_difference = difference,
    expected = expected,
    stringsAsFactors = FALSE
  )
}

set.seed(2001)
n <- 500L
x <- rnorm(n)
z <- rnorm(n)
linear_index <- 0.35 + 0.70 * x - 0.45 * z

binary_inverse <- list(
  logit = plogis,
  probit = pnorm,
  cloglog = function(eta) -expm1(-exp(eta))
)

for (link in names(binary_inverse)) {
  set.seed(2100 + match(link, names(binary_inverse)))
  y <- rbinom(n, 1L, binary_inverse[[link]](linear_index))
  fit <- glm(y ~ x + z, family = binomial(link = link))
  fitted_index <- as.numeric(predict(fit, type = "link"))

  set.seed(2200 + match(link, names(binary_inverse)))
  sure_response <- as.numeric(sure::surrogate(fit, method = "latent"))

  # The local sure development version maps cloglog to its Gumbel-min
  # implementation for every supported model class. This is the ordinal
  # convention, including when the fitted object is a binomial glm.
  sure_link <- switch(
    link,
    logit = "logit",
    probit = "probit",
    cloglog = "cloglog_min"
  )
  set.seed(2200 + match(link, names(binary_inverse)))
  sudo_using_sure_law <- complete_surrogate(
    y + 1L, fitted_index, c(-Inf, 0, Inf), link = sure_link
  )
  implementation_difference <- max(abs(sure_response - sudo_using_sure_law))
  record_check(
    "binary", link, "same law and uniforms",
    implementation_difference, "zero"
  )
  stopifnot(implementation_difference < tolerance)

  sudo_link <- if (link == "cloglog") "cloglog" else link
  set.seed(2200 + match(link, names(binary_inverse)))
  sudo_response <- complete_surrogate(
    y + 1L, fitted_index, c(-Inf, 0, Inf), link = sudo_link
  )
  convention_difference <- max(abs(sure_response - sudo_response))
  record_check(
    "binary", link, "operational SUDO convention",
    convention_difference, if (link == "cloglog") "nonzero" else "zero"
  )
  if (link == "cloglog") {
    stopifnot(convention_difference > 1e-3)

    glm_probability <- binary_inverse$cloglog(fitted_index)
    sure_latent_probability <- exp(-exp(-fitted_index))
    sudo_latent_probability <- 1 - pgumbel(0, loc = fitted_index)
    stopifnot(max(abs(glm_probability - sudo_latent_probability)) < tolerance)
    stopifnot(max(abs(glm_probability - sure_latent_probability)) > 0.1)
  } else {
    stopifnot(convention_difference < tolerance)
  }

  set.seed(2300 + match(link, names(binary_inverse)))
  sure_response <- as.numeric(sure::surrogate(fit, method = "latent"))
  set.seed(2300 + match(link, names(binary_inverse)))
  sure_residual <- sure:::generate_surrogate(
    fit, method = "latent", residual = TRUE
  )
  residual_difference <- max(
    abs(sure_residual - (sure_response - fitted_index))
  )
  record_check(
    "binary", link, "sure residual equals S minus fitted mean",
    residual_difference, "zero"
  )
  stopifnot(residual_difference < tolerance)
}

set.seed(2401)
latent <- 0.55 * x - 0.30 * z + rlogis(n)
ordinal_y <- cut(
  latent,
  breaks = c(-Inf, -1.1, -0.2, 0.7, 1.5, Inf),
  labels = FALSE
)
ordinal_data <- data.frame(Y = ordered(ordinal_y), x = x, z = z)
ordinal_links <- c("logit", "probit", "cloglog")

for (link in ordinal_links) {
  fit <- ordinal::clm(Y ~ x + z, data = ordinal_data, link = link)
  y_code <- as.integer(fit$y)
  sure_mean <- sure:::get_mean_response(fit)
  sure_bounds <- sure:::get_bounds(fit)
  sudo_link <- switch(
    link,
    logit = "logit",
    probit = "probit",
    cloglog = "cloglog_min"
  )

  set.seed(2500 + match(link, ordinal_links))
  sure_response <- as.numeric(sure::surrogate(fit, method = "latent"))
  set.seed(2500 + match(link, ordinal_links))
  sudo_shifted <- complete_surrogate(
    y_code, sure_mean, sure_bounds, link = sudo_link
  )
  shifted_difference <- max(abs(sure_response - sudo_shifted))
  record_check(
    "ordinal", link, "sure threshold-origin coordinates",
    shifted_difference, "zero"
  )
  stopifnot(shifted_difference < tolerance)

  # sure sets the first threshold to zero. Returning to the clm coordinates
  # adds alpha[1] to both the latent mean and every finite threshold.
  threshold_origin <- unname(fit$alpha[1L])
  original_bounds <- sure_bounds
  finite <- is.finite(original_bounds)
  original_bounds[finite] <- original_bounds[finite] + threshold_origin
  set.seed(2500 + match(link, ordinal_links))
  sudo_original <- complete_surrogate(
    y_code, sure_mean + threshold_origin, original_bounds, link = sudo_link
  )
  origin_difference <- max(
    abs(sudo_original - threshold_origin - sure_response)
  )
  record_check(
    "ordinal", link, "original clm coordinates after constant shift",
    origin_difference, "zero"
  )
  stopifnot(origin_difference < tolerance)

  set.seed(2600 + match(link, ordinal_links))
  sure_response <- as.numeric(sure::surrogate(fit, method = "latent"))
  set.seed(2600 + match(link, ordinal_links))
  sure_residual <- sure:::generate_surrogate(
    fit, method = "latent", residual = TRUE
  )
  residual_difference <- max(
    abs(sure_residual - (sure_response - sure_mean))
  )
  record_check(
    "ordinal", link, "sure residual equals S minus fitted mean",
    residual_difference, "zero"
  )
  stopifnot(residual_difference < tolerance)
}

results <- do.call(rbind, rows)
dir.create("R/results", showWarnings = FALSE, recursive = TRUE)
write.csv(
  results,
  "R/results/stage20_sure_equivalence.csv",
  row.names = FALSE
)

print(results, row.names = FALSE)
cat("\nAll Stage 20 assertions passed.\n")
cat(paste(
  "Conclusion: randomized SUDO and sure latent responses are identical",
  "conditional draws for binary logit/probit and ordinal logit/probit/cloglog",
  "after sure's ordinal threshold-origin shift. Binary glm cloglog is the",
  paste0("exception: local sure ", sure_version, " uses Gumbel-min, while the glm link requires"),
  "Gumbel-max.\n"
))
