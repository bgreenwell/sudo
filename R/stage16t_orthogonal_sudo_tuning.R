# Stage 16t: dedicated-seed clipping selection for orthogonal binary SUDO.
#
# Every candidate re-tunes GAMSEL inside every outer fold. Selection minimizes
# worst-cell target bias on the two primary cells. Confirmatory stage 16 uses
# disjoint seeds.
#
# Run from the repository root:
#   Rscript R/stage16t_orthogonal_sudo_tuning.R
#
# Smoke test:
#   SUDO_STAGE16_SMOKE=1 Rscript R/stage16t_orthogonal_sudo_tuning.R

source("R/stage16_orthogonal_support.R")
stage16_require_packages()
stage16_unit_checks()

smoke <- nzchar(Sys.getenv("SUDO_STAGE16_SMOKE", unset = ""))
n_reps <- stage16_env_int("SUDO_STAGE16T_REPS", if (smoke) 2L else 20L)
outer_folds <- stage16_env_int("SUDO_STAGE16T_OUTER_FOLDS", if (smoke) 2L else 5L)
inner_folds <- stage16_env_int("SUDO_STAGE16T_INNER_FOLDS", if (smoke) 4L else 5L)
cores <- stage16_env_int("SUDO_STAGE16T_CORES", if (smoke) 1L else 4L)
chunk_size <- stage16_env_int(
  "SUDO_STAGE16T_CHUNK_SIZE", if (smoke) 1L else 5L
)
clips <- c(1e-3, 1e-4, 1e-5, 1e-6)
clip_filter <- Sys.getenv("SUDO_STAGE16T_CLIP", unset = "")
if (nzchar(clip_filter)) clips <- as.numeric(strsplit(clip_filter, ",")[[1]])
cells <- stage16_cells(smoke)
cells <- cells[cells$primary, , drop = FALSE]
cell_filter <- Sys.getenv("SUDO_STAGE16T_CELL", unset = "")
if (nzchar(cell_filter)) {
  cells <- cells[cells$cell %in% strsplit(cell_filter, ",")[[1]], , drop = FALSE]
}
stopifnot(nrow(cells) >= 1L, length(clips) >= 1L, all(clips > 0),
          all(clips < 0.5), n_reps >= 2L, outer_folds >= 2L,
          inner_folds >= 4L, cores >= 1L, chunk_size >= 1L)

expected_primary_cells <- stage16_cells(FALSE)
expected_primary_cells <- expected_primary_cells[
  expected_primary_cells$primary, "cell"
]
full_run <- !smoke && n_reps >= 20L && outer_folds >= 5L &&
  inner_folds >= 5L && identical(as.character(cells$cell),
                                  expected_primary_cells) &&
  identical(sort(clips), sort(c(1e-3, 1e-4, 1e-5, 1e-6)))
suffix <- if (full_run) "" else "_smoke"
checkpoint_file <- paste0(
  "R/results/stage16t_orthogonal_sudo_checkpoint", suffix, ".csv"
)
checkpoint <- if (file.exists(checkpoint_file)) {
  read.csv(checkpoint_file, stringsAsFactors = FALSE)
} else {
  data.frame()
}
checkpoint_keys <- c(
  "cell", "clip", "cell_n", "cell_p", "outer_folds", "inner_folds",
  "replication"
)
if (nrow(checkpoint) && !all(checkpoint_keys %in% names(checkpoint))) {
  checkpoint <- data.frame()
}
if (nrow(checkpoint)) {
  cell_match <- match(checkpoint$cell, cells$cell)
  checkpoint <- checkpoint[
    !is.na(cell_match) & checkpoint$clip %in% clips &
      checkpoint$cell_n == cells$n[cell_match] &
      checkpoint$cell_p == cells$p[cell_match] &
      checkpoint$outer_folds == outer_folds &
      checkpoint$inner_folds == inner_folds, , drop = FALSE
  ]
}

cluster <- if (cores > 1L) stage16_cluster(cores) else NULL
for (cell_index in seq_len(nrow(cells))) {
  cell <- cells[cell_index, ]
  for (clip in clips) {
    completed <- if (nrow(checkpoint)) {
      unique(checkpoint$replication[
        checkpoint$cell == cell$cell & checkpoint$clip == clip &
          checkpoint$replication <= n_reps
      ])
    } else {
      integer()
    }
    remaining <- setdiff(seq_len(n_reps), completed)
    chunks <- split(remaining, ceiling(seq_along(remaining) / chunk_size))
    worker <- function(replication) {
      seed <- 1602000L + cell_index * 10000L + replication
      stage16_one_rep(
        seed, cell, clip, outer_folds, inner_folds,
        include_one_se = FALSE
      )
    }
    for (chunk in chunks) {
      if (!is.null(cluster)) {
        parallel::clusterExport(
          cluster,
          c("cell", "clip", "cell_index", "outer_folds", "inner_folds",
            "worker"), envir = environment()
        )
      }
      block <- do.call(rbind, if (is.null(cluster)) {
        lapply(chunk, worker)
      } else {
        parallel::parLapplyLB(cluster, chunk, worker)
      })
      block$cell <- cell$cell
      block$primary <- cell$primary
      block$clip <- clip
      block$cell_n <- cell$n
      block$cell_p <- cell$p
      block$outer_folds <- outer_folds
      block$inner_folds <- inner_folds
      block$replication <- chunk
      checkpoint <- rbind(checkpoint, block)
      dir.create("R/results", showWarnings = FALSE)
      write.csv(checkpoint, checkpoint_file, row.names = FALSE)
      cat(sprintf(
        "%-12s clip %.0e checkpointed %d/%d\n",
        cell$cell, clip,
        length(unique(checkpoint$replication[
          checkpoint$cell == cell$cell & checkpoint$clip == clip
        ])), n_reps
      ))
    }
  }
}
if (!is.null(cluster)) parallel::stopCluster(cluster)
cell_match <- match(checkpoint$cell, cells$cell)
replications <- checkpoint[
  !is.na(cell_match) & checkpoint$clip %in% clips &
    checkpoint$cell_n == cells$n[cell_match] &
    checkpoint$cell_p == cells$p[cell_match] &
    checkpoint$outer_folds == outer_folds &
    checkpoint$inner_folds == inner_folds &
    checkpoint$replication <= n_reps, , drop = FALSE
]

# Every clipping arm must use the same samples and outer folds.
signature <- aggregate(
  cbind(sample_signature, fold_signature) ~ cell + replication,
  replications, function(value) length(unique(value))
)
stopifnot(all(signature$sample_signature == 1L),
          all(signature$fold_signature == 1L),
          all(replications$direct_contrast_spread < 1e-8))

cell_summary <- do.call(rbind, lapply(
  split(replications, list(replications$clip, replications$cell), drop = TRUE),
  function(block) data.frame(
    clip = block$clip[1], cell = block$cell[1], n_reps = nrow(block),
    bias = mean(block$orthogonal - block$truth),
    mc_se_bias = sd(block$orthogonal - block$truth) / sqrt(nrow(block)),
    sd = sd(block$orthogonal),
    mean_clipping_rate = mean(block$clipping_rate),
    max_inverse_information = max(block$max_inverse_information),
    fail_rate = mean(!is.finite(block$orthogonal))
  )
))
rownames(cell_summary) <- NULL
selection <- do.call(rbind, lapply(split(cell_summary, cell_summary$clip),
                                   function(block) data.frame(
  clip = block$clip[1],
  worst_abs_bias = max(abs(block$bias)),
  mean_abs_bias = mean(abs(block$bias)),
  max_clipping_rate = max(block$mean_clipping_rate),
  max_fail_rate = max(block$fail_rate)
)))
rownames(selection) <- NULL
best_value <- min(selection$worst_abs_bias)
eligible <- selection[
  selection$worst_abs_bias <= best_value + 0.005, , drop = FALSE
]
selected <- eligible[which.max(eligible$clip), , drop = FALSE]
selection$selected <- selection$clip == selected$clip
selection <- selection[order(selection$worst_abs_bias, -selection$clip), ]
print(cell_summary[order(cell_summary$clip, cell_summary$cell), ],
      row.names = FALSE)
cat("\nClipping selection:\n")
print(selection, row.names = FALSE)

if (full_run) {
  stopifnot(nrow(selected) == 1L, is.finite(selected$clip),
            all(selection$max_fail_rate < 0.02))
  cat(sprintf(
    "PASS: clipping %.0e is frozen on dedicated stage-16t seeds\n",
    selected$clip
  ))
} else {
  cat("SMOKE PASS: a frozen clipping rule requires the documented full run\n")
}

dir.create("R/results", showWarnings = FALSE)
write.csv(
  replications,
  paste0("R/results/stage16t_orthogonal_sudo_replications", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  cell_summary,
  paste0("R/results/stage16t_orthogonal_sudo_tuning", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  selection,
  paste0("R/results/stage16t_orthogonal_sudo_selection", suffix, ".csv"),
  row.names = FALSE
)
