# Is the pin effect bigger than the noise between categories?
#
# The repository reports the ablation as a difference of two means: PatchCore
# CAR 0.415 with the training confound rate left free, 0.189 with it pinned. No
# test, no interval, nothing that says whether that gap would survive a
# different draw of MVTec categories. The whole argument of section 5 rests on
# it, so it is worth asking properly.
#
# The design is paired: every run at pin on has a run at pin off with the same
# category, the same seed and the same detector, so the difference is taken
# within the pair and the between category spread cancels out of the point
# estimate. What it does not cancel out of is the uncertainty, which is why the
# interval below resamples whole categories rather than runs.
#
# Two things are checked, deterministic and stochastic:
#
#   the cell means, which must reproduce reports/real_mechanism_summary.csv
#   the pin effect, which must be a real difference and not category noise
#
# and one control: at rho = 0 the natural training rate already equals the
# pinned rate, so pinning is a no-op there and every paired difference must be
# exactly zero. A test that cannot return a null result is not a test.
#
# Base R, no packages, so CI needs nothing beyond r-base-core.
#
# Run: Rscript verify/verify.R <repo root>

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

TOL <- 6e-5        # half of the last published decimal place, plus headroom
DRAWS <- 20000
ALPHA <- 0.01
failures <- 0

# The JSON in reports/ is what json.dumps writes with indent=2: an array of flat
# objects, one scalar per line. Reading it by line keeps this package free.
read_runs <- function(path) {
    lines <- readLines(path, warn = FALSE)
    recs <- list()
    cur <- NULL
    for (ln in lines) {
        t <- trimws(ln)
        if (t == "{") { cur <- list(); next }
        if (substr(t, 1, 1) == "}") {
            if (!is.null(cur)) { recs[[length(recs) + 1]] <- cur; cur <- NULL }
            next
        }
        if (is.null(cur)) next
        m <- regmatches(t, regexec('^"([^"]+)"[[:space:]]*:[[:space:]]*(.*)$', t))[[1]]
        if (length(m) != 3) next
        key <- m[2]
        val <- sub(",$", "", m[3])
        cur[[key]] <- if (substr(val, 1, 1) == '"') gsub('^"|"$', "", val)
                      else if (val == "true") TRUE
                      else if (val == "false") FALSE
                      else as.numeric(val)
    }
    if (length(recs) == 0) stop(paste("no records in", path))
    do.call(rbind, lapply(recs, function(r) as.data.frame(r, stringsAsFactors = FALSE)))
}

runs <- read_runs(file.path(root, "reports", "real_mechanism.json"))
pub <- read.csv(file.path(root, "reports", "real_mechanism_summary.csv"),
                stringsAsFactors = FALSE)
cat(sprintf("read %d runs from real_mechanism.json, %d rows from the summary\n\n",
            nrow(runs), nrow(pub)))

# --- the deterministic half -------------------------------------------------

cat("cell means, against reports/real_mechanism_summary.csv\n")
worst <- 0
cells <- 0
for (i in seq_len(nrow(pub))) {
    sel <- runs$detector == pub$detector[i] &
           runs$pinned_train_rate == (pub$pinned_train_rate[i] == "True") &
           runs$rho == pub$rho[i]
    if (sum(sel) == 0) {
        cat(sprintf("  FAIL no runs for %s pin=%s rho=%s\n",
                    pub$detector[i], pub$pinned_train_rate[i], pub$rho[i]))
        failures <- failures + 1
        next
    }
    for (col in c("car", "peak_on_defect", "auroc")) {
        got <- mean(runs[[if (col == "peak_on_defect") "peak_on_defect" else col]][sel])
        d <- abs(got - pub[[col]][i])
        worst <- max(worst, d)
        cells <- cells + 1
        if (d > TOL) {
            cat(sprintf("  FAIL %s pin=%s rho=%s %s: R %.10f, published %.4f\n",
                        pub$detector[i], pub$pinned_train_rate[i], pub$rho[i], col,
                        got, pub[[col]][i]))
            failures <- failures + 1
        }
    }
}
cat(sprintf("  %d cells reproduced, max |d| %.2e\n", cells, worst))

# --- the pin effect ---------------------------------------------------------

# One paired difference per category and seed: CAR with the rate free minus CAR
# with it pinned, everything else held.
paired <- function(det, rho) {
    free <- runs[runs$detector == det & runs$rho == rho & !runs$pinned_train_rate, ]
    pin <- runs[runs$detector == det & runs$rho == rho & runs$pinned_train_rate, ]
    key <- function(d) paste(d$category, d$seed)
    idx <- match(key(free), key(pin))
    if (any(is.na(idx))) stop("a run at pin off has no partner at pin on")
    data.frame(category = free$category, diff = free$car - pin$car[idx],
               stringsAsFactors = FALSE)
}

# Resampling categories, not runs. Five MVTec categories were chosen; the
# question the README's number invites is whether another five would say the
# same, and that is the unit the interval has to respect.
cluster_ci <- function(p) {
    per <- tapply(p$diff, p$category, mean)
    k <- length(per)
    stats <- replicate(DRAWS, mean(per[sample.int(k, k, replace = TRUE)]))
    quantile(stats, c(0.025, 0.975), names = FALSE)
}

cat("\nthe pin effect on CAR, paired within category and seed\n")
for (det in c("patchcore", "padim")) {
    for (rho in c(0.0, 1.0)) {
        p <- paired(det, rho)
        d <- p$diff
        published <- pub$car[pub$detector == det & pub$pinned_train_rate == "False" &
                             pub$rho == rho] -
                     pub$car[pub$detector == det & pub$pinned_train_rate == "True" &
                             pub$rho == rho]
        if (all(d == 0)) {
            # The control. Pinning at rho = 0 changes nothing, and it must show.
            ok <- abs(published) < TOL
            failures <- failures + !ok
            cat(sprintf("  %-9s rho=%.1f  n=%2d  every paired difference exactly 0, "
                        , det, rho, length(d)))
            cat(sprintf("published gap %.4f  %s\n", published, if (ok) "ok" else "FAIL"))
            next
        }
        tt <- t.test(d)
        ci <- cluster_ci(p)
        agrees <- abs(mean(d) - published) < TOL
        strong <- tt$p.value < ALPHA
        excludes <- ci[1] > 0
        ok <- agrees && strong && excludes
        failures <- failures + !ok
        cat(sprintf("  %-9s rho=%.1f  n=%2d  mean %+.4f (published %+.4f)  "
                    , det, rho, length(d), mean(d), published))
        cat(sprintf("t=%.2f p=%.1e  category cluster CI [%.4f, %.4f]  %s\n",
                    tt$statistic, tt$p.value, ci[1], ci[2], if (ok) "ok" else "FAIL"))
        if (!agrees) cat("      FAIL the paired mean is not the published gap\n")
        if (!strong) cat(sprintf("      FAIL p is not below %.2f\n", ALPHA))
        if (!excludes) cat("      FAIL the category cluster interval contains zero\n")
    }
}

# --- the resnet18 sweep against its own null --------------------------------

# The README says the resnet18 CAR never reaches its random-heatmap null. That
# is an ordering of two published means. Every run carries its own null, so the
# comparison can be made paired, run by run, which says how much of a separation
# it actually is.
cat("\nresnet18 CAR against its own random-heatmap null, paired run by run\n")
back <- read_runs(file.path(root, "reports", "real_backbone.json"))
for (det in c("patchcore", "padim")) {
    for (rho in sort(unique(back$rho))) {
        sel <- back$detector == det & back$rho == rho
        d <- back$car[sel] - back$car_random[sel]
        tt <- t.test(d, alternative = "less")
        below <- mean(d) < 0
        # Below 0.75 the separation is required to be significant. At rho = 1
        # the README claims an ordering of the means and nothing more, so that
        # is all this insists on, and the measured t is printed either way.
        ok <- below && (rho >= 1.0 || tt$p.value < ALPHA)
        failures <- failures + !ok
        cat(sprintf("  %-9s rho=%.2f  n=%2d  mean CAR minus null %+.4f  t=%+.2f  "
                    , det, rho, length(d), mean(d), tt$statistic))
        cat(sprintf("p=%.1e  %s\n", tt$p.value, if (ok) "ok" else "FAIL"))
    }
}

cat("\n")
if (failures > 0) {
    cat(sprintf("%d checks failed\n", failures))
    quit(status = 1)
}
cat("R reproduces the cell means and finds the pin effect larger than the\n")
cat("spread between categories\n")
