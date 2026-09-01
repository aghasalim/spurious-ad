//! Exact tests, by enumeration rather than sampling.
//!
//! The R next door answers two questions with a t-test and a 20,000 draw
//! bootstrap. Both are approximations: the t-test assumes the paired
//! differences are normal, which for fifteen numbers with three exact zeros in
//! them is an assumption and not a fact, and a sampled bootstrap interval
//! carries its own Monte Carlo error that nobody ever measured.
//!
//! Neither approximation is necessary at this size. The design is paired with
//! fifteen pairs, so the sign-flip permutation distribution has 2^15 = 32,768
//! points and can be walked in full: the p value that comes out is exact, with
//! no distributional assumption and no sampling error. The cluster bootstrap
//! resamples five categories, so it has 5^5 = 3,125 points and can also be
//! walked in full: the interval that comes out is the exact resampling
//! interval, not an estimate of it.
//!
//! That is the part the Python could not afford to write in a numpy loop and
//! the reason this is in Rust: 24 cells x 35,893 enumerated statistics.
//!
//! Two claims are put through it:
//!
//!   the pin effect     reports/real_mechanism.json, CAR with the training
//!                      confound rate free minus CAR with it pinned
//!   CAR under its null reports/real_backbone.json, CAR minus the random
//!                      heatmap control for the same run
//!
//! Run: cd verify/permtest && cargo run --release --quiet -- <repo root>

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process::exit;

const TOL: f64 = 6e-5; // half of the last published decimal place, plus headroom
const ALPHA: f64 = 0.01;

#[derive(Clone, Debug)]
struct Run {
    category: String,
    detector: String,
    rho: f64,
    seed: i64,
    pinned: bool,
    car: f64,
    car_random: f64,
}

/// The JSON in reports/ is what json.dumps writes with indent=2: an array of
/// flat objects, one scalar per line. Reading it by line keeps this crate free
/// of dependencies, and a change to the layout fails here loudly.
fn load(path: &str) -> Vec<Run> {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| { eprintln!("cannot read {}: {}", path, e); exit(2) });

    let mut runs = Vec::new();
    let mut cur: Option<Run> = None;

    for line in text.lines() {
        let t = line.trim();
        if t == "{" {
            cur = Some(Run { category: String::new(), detector: String::new(), rho: 0.0,
                             seed: -1, pinned: false, car: f64::NAN, car_random: f64::NAN });
            continue;
        }
        if t.starts_with('}') {
            if let Some(r) = cur.take() {
                runs.push(r);
            }
            continue;
        }
        let r = match cur.as_mut() { Some(r) => r, None => continue };

        let mut parts = t.splitn(2, ':');
        let key = match parts.next() { Some(k) => k.trim().trim_matches('"'), None => continue };
        let raw = match parts.next() { Some(v) => v.trim().trim_end_matches(','), None => continue };
        let val = raw.trim_matches('"');

        match key {
            "category" => r.category = val.to_string(),
            "detector" => r.detector = val.to_string(),
            "rho" => r.rho = val.parse().unwrap_or(f64::NAN),
            "seed" => r.seed = val.parse().unwrap_or(-1),
            "pinned_train_rate" => r.pinned = raw == "true",
            "car" => r.car = val.parse().unwrap_or(f64::NAN),
            "car_random" => r.car_random = val.parse().unwrap_or(f64::NAN),
            _ => {}
        }
    }
    if runs.is_empty() {
        eprintln!("{} holds no records", path);
        exit(2);
    }
    runs
}

/// Exact two-sided paired permutation p value. Under the null the sign of each
/// paired difference is arbitrary, so every one of the 2^n sign assignments is
/// equally likely; the p value is the share of them whose mean is at least as
/// far from zero as the one observed. Exhaustive, so this is the p value and
/// not an estimate of it.
fn exact_sign_p(d: &[f64]) -> f64 {
    let n = d.len();
    assert!(n <= 24, "2^n enumeration is only sane for small n");
    let observed: f64 = d.iter().sum::<f64>().abs();
    // A difference of exactly zero flips to zero, so it doubles the count of
    // assignments that tie the observed statistic rather than changing it. The
    // enumeration handles that on its own, which is why it is done in full.
    let total = 1u64 << n;
    let mut at_least = 0u64;
    for mask in 0..total {
        let mut s = 0.0;
        for (i, &x) in d.iter().enumerate() {
            s += if mask >> i & 1 == 1 { x } else { -x };
        }
        if s.abs() >= observed - 1e-12 {
            at_least += 1;
        }
    }
    at_least as f64 / total as f64
}

/// numpy's default quantile, linear interpolation between order statistics.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi { sorted[lo] } else { sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo]) }
}

/// The complete cluster bootstrap distribution: every one of the k^k ways to
/// draw k categories from k with replacement, each giving the mean of the
/// resampled category means. No sampling, so no Monte Carlo error.
fn exhaustive_cluster_ci(per_category: &[f64]) -> (f64, f64, usize) {
    let k = per_category.len();
    assert!(k <= 8, "k^k enumeration is only sane for small k");
    let total = (k as u64).pow(k as u32);
    let mut stats = Vec::with_capacity(total as usize);
    for draw in 0..total {
        let mut d = draw;
        let mut s = 0.0;
        for _ in 0..k {
            s += per_category[(d % k as u64) as usize];
            d /= k as u64;
        }
        stats.push(s / k as f64);
    }
    stats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (quantile(&stats, 0.025), quantile(&stats, 0.975), stats.len())
}

fn per_category_means(pairs: &[(String, f64)]) -> Vec<f64> {
    let mut by: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    for (cat, d) in pairs {
        let e = by.entry(cat.as_str()).or_insert((0.0, 0));
        e.0 += d;
        e.1 += 1;
    }
    by.values().map(|(s, n)| s / *n as f64).collect()
}

fn published_gap(root: &str, detector: &str, rho: f64) -> Option<f64> {
    let text = fs::read_to_string(format!("{}/reports/real_mechanism_summary.csv", root)).ok()?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next()?.split(',').collect();
    let idx = |name: &str| header.iter().position(|h| *h == name);
    let (d, p, r, c) = (idx("detector")?, idx("pinned_train_rate")?, idx("rho")?, idx("car")?);
    let (mut free, mut pinned) = (None, None);
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() <= c || f[d] != detector { continue; }
        if f[r].parse::<f64>().ok()? != rho { continue; }
        let car: f64 = f[c].parse().ok()?;
        if f[p] == "True" { pinned = Some(car) } else { free = Some(car) }
    }
    Some(free? - pinned?)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let root = args.get(1).map(String::as_str).unwrap_or(".");
    let mut failures = 0;
    let mut enumerated: u64 = 0;

    // --- the pin effect ------------------------------------------------------

    let mech = load(&format!("{}/reports/real_mechanism.json", root));
    let mut rhos: Vec<f64> = mech.iter().map(|r| r.rho).collect();
    rhos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rhos.dedup();

    println!("the pin effect on CAR, exact paired permutation over 2^n sign assignments");
    println!("and the complete k^k category cluster bootstrap\n");
    for detector in ["patchcore", "padim"] {
        for &rho in &rhos {
            let free: Vec<&Run> = mech.iter()
                .filter(|r| r.detector == detector && r.rho == rho && !r.pinned).collect();
            let pinned: Vec<&Run> = mech.iter()
                .filter(|r| r.detector == detector && r.rho == rho && r.pinned).collect();
            let mut pairs: Vec<(String, f64)> = Vec::new();
            for f in &free {
                match pinned.iter().find(|p| p.category == f.category && p.seed == f.seed) {
                    Some(p) => pairs.push((f.category.clone(), f.car - p.car)),
                    None => {
                        println!("  FAIL {} rho={} {} seed {} has no partner at pin on",
                                 detector, rho, f.category, f.seed);
                        failures += 1;
                    }
                }
            }
            let d: Vec<f64> = pairs.iter().map(|(_, x)| *x).collect();
            let n = d.len();
            let mean = d.iter().sum::<f64>() / n as f64;
            let p = exact_sign_p(&d);
            let per = per_category_means(&pairs);
            let (lo, hi, draws) = exhaustive_cluster_ci(&per);
            enumerated += (1u64 << n) + draws as u64;

            let gap = published_gap(root, detector, rho).unwrap_or(f64::NAN);
            let agrees = (mean - gap).abs() < TOL;

            // At rho = 0 the natural training rate already equals the pinned
            // rate, so pinning is a no-op and the exact p must come back at 1.
            // A test that cannot return a null result is not a test.
            let ok = if rho == 0.0 {
                agrees && d.iter().all(|x| *x == 0.0) && p == 1.0
            } else if rho >= 1.0 {
                agrees && p < ALPHA && lo > 0.0
            } else {
                agrees
            };
            failures += !ok as i32;

            println!("  {:<9} rho={:.1}  n={:2}  mean {:+.4} (published {:+.4})  \
                      exact p {:.3e} over {} assignments  cluster CI [{:+.4}, {:+.4}] \
                      over {} resamples  {}",
                     detector, rho, n, mean, gap, p, 1u64 << n, lo, hi, draws,
                     if ok { "ok" } else { "FAIL" });
        }
    }

    // --- CAR against its own null on the resnet18 sweep ----------------------

    println!("\nresnet18 CAR minus its random-heatmap null, exact paired permutation\n");
    let back = load(&format!("{}/reports/real_backbone.json", root));
    let mut brhos: Vec<f64> = back.iter().map(|r| r.rho).collect();
    brhos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    brhos.dedup();
    for detector in ["patchcore", "padim"] {
        for &rho in &brhos {
            let d: Vec<f64> = back.iter()
                .filter(|r| r.detector == detector && r.rho == rho)
                .map(|r| r.car - r.car_random).collect();
            let n = d.len();
            let mean = d.iter().sum::<f64>() / n as f64;
            let p = exact_sign_p(&d);
            enumerated += 1u64 << n;
            // The README claims CAR stays below the null. Below rho = 1 that is
            // required to be a real separation; at rho = 1 the claim is an
            // ordering of the published means and nothing more, so that is all
            // this insists on. The exact p is printed either way.
            let ok = mean < 0.0 && (rho >= 1.0 || p < ALPHA);
            failures += !ok as i32;
            println!("  {:<9} rho={:.2}  n={:2}  mean {:+.4}  exact p {:.3e}  {}",
                     detector, rho, n, mean, p, if ok { "ok" } else { "FAIL" });
        }
    }

    println!();
    if failures > 0 {
        println!("{} exact checks failed", failures);
        exit(1);
    }
    println!("Rust enumerated {} statistics in full, so every p value and every interval",
             enumerated);
    println!("above is exact: no normal approximation, no Monte Carlo error");
}
