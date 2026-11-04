#!/usr/bin/env Rscript
#
# Track B of the analysis: repeated-measures ANOVA and the Tukey post-hoc test.
#
# The seeds are the subjects; every k level (configuration) runs on every seed.
#
#   omnibus     aov(value ~ config + Error(seed/config)): F, df1, df2 and the
#               uncorrected p-value of the seed:config stratum
#   sphericity  Mauchly's test; the Greenhouse-Geisser and Huynh-Feldt epsilons
#               and corrected p-values, computed from the covariance of the
#               orthonormal contrasts and checked against
#               anova(lm(Y ~ 1), X = ~1, test = "Spherical") (the script stops
#               on a difference above 1e-8)
#   post-hoc    emmeans on the aov model, all pairs, Tukey adjustment, pooled
#               error of the seed:config stratum
#
# Sign rule: every difference is the later k level minus the earlier k level in
# the given order. With the ground truth first, a ground-truth row is k level
# minus ground truth, so a positive difference means the k level scores higher.
# A pair is separated when its Tukey-adjusted p-value is below alpha. The
# confidence intervals of the pairs have the level 1 - alpha and the Tukey
# adjustment.
#
# Usage:
#   Rscript track_b.R <long.csv> <output folder> <alpha> <level1,level2,...>
# The long table has the columns seed, config, value, with exactly one row for
# each seed and k level. Output files: track_b_omnibus.csv,
# track_b_pairwise.csv, r_session.txt.
#
# Without residual variance (for example the same difference between two k
# levels in every seed) the tests have no value: the script writes empty values
# (NA) and the note "no residual variance", and it does not stop.
#
# Requires the R package emmeans.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript track_b.R <long.csv> <output folder> <alpha> <level1,level2,...>")
}
input_path <- args[1]
output_dir <- args[2]
alpha <- as.numeric(args[3])
levels_order <- strsplit(args[4], ",", fixed = TRUE)[[1]]
if (!is.finite(alpha) || alpha <= 0 || alpha >= 1) stop("alpha must be above 0 and below 1")
if (!requireNamespace("emmeans", quietly = TRUE)) {
  stop("The R package emmeans is not installed. Install it in R with install.packages(\"emmeans\").")
}
suppressMessages(library(emmeans))
if (!dir.exists(output_dir)) stop(sprintf("The output folder does not exist: %s", output_dir))

long_data <- read.csv(input_path, stringsAsFactors = FALSE)
missing_cols <- setdiff(c("seed", "config", "value"), names(long_data))
if (length(missing_cols) > 0) stop(sprintf("Missing columns: %s", paste(missing_cols, collapse = ", ")))
if (!setequal(unique(long_data$config), levels_order) || anyDuplicated(levels_order)) {
  stop("The k levels of the long table differ from the given order")
}
long_data$seed <- factor(long_data$seed)
long_data$config <- factor(long_data$config, levels = levels_order)
counts <- table(long_data$seed, long_data$config)
if (any(counts != 1)) stop("The design must have exactly one row for each seed and k level")
n <- nlevels(long_data$seed)
m <- length(levels_order)
if (n < 3 || m < 2) stop("Track B needs 3 or more seeds and 2 or more k levels")
cat(sprintf("Track B (R): %d seeds, %d k levels\n", n, m))

# Omnibus test
model <- aov(value ~ config + Error(seed / config), data = long_data)
stratum <- as.data.frame(summary(model)[["Error: seed:config"]][[1]])
rownames(stratum) <- trimws(rownames(stratum))
f_stat <- stratum["config", "F value"]
df1 <- stratum["config", "Df"]
df2 <- stratum["Residuals", "Df"]
p_uncorrected <- stratum["config", "Pr(>F)"]
total_ss <- sum((long_data$value - mean(long_data$value))^2)
residual_ss <- stratum["Residuals", "Sum Sq"]
degenerate <- !is.finite(f_stat) || residual_ss <= 1e-12 * max(total_ss, .Machine$double.xmin)

# Sphericity: wide matrix (seeds x k levels), orthonormal contrasts
wide <- tapply(long_data$value, list(long_data$seed, long_data$config), identity)
wide <- matrix(as.numeric(wide), nrow = n, dimnames = dimnames(wide))
p <- m - 1
contrasts_matrix <- contr.helmert(m)
contrasts_matrix <- sweep(contrasts_matrix, 2, sqrt(colSums(contrasts_matrix^2)), "/")
s_matrix <- t(contrasts_matrix) %*% cov(wide) %*% contrasts_matrix
n_res <- n - 1
notes <- character(0)
mauchly_w <- NA_real_
mauchly_p <- NA_real_
gg_epsilon <- NA_real_
hf_epsilon <- NA_real_
p_gg <- NA_real_
p_hf <- NA_real_
if (degenerate) {
  f_stat <- NA_real_
  p_uncorrected <- NA_real_
  notes <- c(notes, "no residual variance: no F test, no sphericity test, no Tukey test")
} else {
  gg_epsilon <- sum(diag(s_matrix))^2 / (p * sum(s_matrix * s_matrix))
  hf_epsilon <- ((n_res + 1) * p * gg_epsilon - 2) / (p * (n_res - p * gg_epsilon))
  p_gg <- pf(f_stat, gg_epsilon * df1, gg_epsilon * df2, lower.tail = FALSE)
  p_hf <- pf(f_stat, min(1, hf_epsilon) * df1, min(1, hf_epsilon) * df2, lower.tail = FALSE)
  if (n_res >= p) {
    mauchly <- mauchly.test(lm(wide ~ 1), X = ~1)
    mauchly_w <- unname(mauchly$statistic)
    mauchly_p <- mauchly$p.value
  } else {
    notes <- c(notes, "fewer seeds than k levels: no Mauchly test")
  }
  same <- function(a, b) (is.na(a) && is.na(b)) || (is.finite(a) && is.finite(b) && abs(a - b) <= 1e-8 * max(1, abs(a)))
  check_row <- anova(lm(wide ~ 1), X = ~1, test = "Spherical")[1, ]
  if (!same(f_stat, check_row[["F"]]) || !same(p_gg, check_row[["G-G Pr"]]) || !same(p_hf, check_row[["H-F Pr"]])) {
    stop(sprintf("Cross-check with anova.mlm failed: F %.12g vs %.12g, G-G p %.12g vs %.12g, H-F p %.12g vs %.12g",
                 f_stat, check_row[["F"]], p_gg, check_row[["G-G Pr"]], p_hf, check_row[["H-F Pr"]]))
  }
  cat(sprintf("F(%g, %g) = %.4f, p = %.4g; Greenhouse-Geisser epsilon %.4f, p = %.4g; Huynh-Feldt epsilon %.4f, p = %.4g\n",
              df1, df2, f_stat, p_uncorrected, gg_epsilon, p_gg, hf_epsilon, p_hf))
  cat("Cross-check with anova.mlm (test = \"Spherical\"): passed\n")
}
note <- paste(notes, collapse = "; ")
omnibus_significant_gg <- is.finite(p_gg) && p_gg < alpha

omnibus <- data.frame(
  n_seeds = n, n_configs = m, f_stat = f_stat, df1 = df1, df2 = df2, p_uncorrected = p_uncorrected,
  mauchly_w = mauchly_w, mauchly_p = mauchly_p, gg_epsilon = gg_epsilon, p_gg = p_gg,
  hf_epsilon = hf_epsilon, p_hf = p_hf, alpha = alpha, omnibus_significant_gg = omnibus_significant_gg,
  note = note
)
write.csv(omnibus, file.path(output_dir, "track_b_omnibus.csv"), row.names = FALSE)

# Post-hoc test: all pairs, the later k level minus the earlier k level
if (degenerate) {
  index <- t(combn(m, 2))
  means <- colMeans(wide)[levels_order]
  pairwise <- data.frame(
    group1 = levels_order[index[, 1]], group2 = levels_order[index[, 2]],
    diff = unname(means[index[, 2]] - means[index[, 1]]), se = NA_real_, df = df2, t_ratio = NA_real_,
    p_tukey = NA_real_, ci_level = 1 - alpha, lower = NA_real_, upper = NA_real_, reject = NA,
    omnibus_significant_gg = omnibus_significant_gg, stringsAsFactors = FALSE
  )
} else {
  emm <- emmeans(model, ~ config)
  pairs_summary <- summary(pairs(emm, adjust = "tukey", reverse = TRUE), infer = c(TRUE, TRUE),
                           level = 1 - alpha)
  labels <- as.character(pairs_summary$contrast)
  parts <- strsplit(labels, " - ", fixed = TRUE)
  if (any(lengths(parts) != 2)) stop("Unexpected contrast labels from emmeans")
  later <- vapply(parts, `[`, "", 1)
  earlier <- vapply(parts, `[`, "", 2)
  if (!all(later %in% levels_order) || !all(earlier %in% levels_order) ||
      any(match(later, levels_order) <= match(earlier, levels_order))) {
    stop("Unexpected contrast labels or order from emmeans")
  }
  pairwise <- data.frame(
    group1 = earlier, group2 = later, diff = pairs_summary$estimate, se = pairs_summary$SE,
    df = pairs_summary$df, t_ratio = pairs_summary$t.ratio, p_tukey = pairs_summary$p.value,
    ci_level = 1 - alpha, lower = pairs_summary$lower.CL, upper = pairs_summary$upper.CL,
    reject = pairs_summary$p.value < alpha, omnibus_significant_gg = omnibus_significant_gg,
    stringsAsFactors = FALSE
  )
}
pairwise <- pairwise[order(match(pairwise$group1, levels_order), match(pairwise$group2, levels_order)), ]
write.csv(pairwise, file.path(output_dir, "track_b_pairwise.csv"), row.names = FALSE)
cat(sprintf("Pairs: %d, separated (Tukey p below %g): %d\n", nrow(pairwise), alpha, sum(pairwise$reject, na.rm = TRUE)))

writeLines(c(sprintf("R %s", paste(R.version$major, R.version$minor, sep = ".")),
             sprintf("emmeans %s", as.character(packageVersion("emmeans")))),
           file.path(output_dir, "r_session.txt"))
