#!/usr/bin/env bash
# Recompute the published tables in every language here and require agreement.
#
# Every number in the README, in notes/METHODS.md and in every figure comes out
# of one pandas groupby, in experiments/sweep.py, experiments/mechanism.py and
# experiments/real.py. If one of those aggregations were wrong, nothing
# downstream would notice, because everything downstream reads the same output.
# The tests check that the generator and the metric are right, not that the
# tables are.
#
# So the tables are recomputed from the raw per-run JSON by seven independent
# implementations in seven languages, and a mistake would have to be made
# identically in all of them to survive.
#
# Each is skipped with a clear message if its toolchain is absent, so this runs
# on a laptop with only some of them. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

tmp="${TMPDIR:-/tmp}"
pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then
        printf -- '--- %s: ok\n' "$name"; pass=$((pass + 1))
    else
        printf -- '--- %s: FAILED\n' "$name"; fail=$((fail + 1))
    fi
}

# SQLite has no assertion of its own, so the script prints a line beginning FAIL
# for every disagreement and this turns that into an exit status.
check_sql () {
    local out
    out=$(sqlite3 -init verify/summaries.sql :memory: "" 2>&1)
    printf '%s\n' "$out"
    if printf '%s' "$out" | grep -q '^FAIL'; then return 1; fi
    printf '%s' "$out" | grep -q '^SQL reproduces'
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "$tmp/spad_summary" verify/summary.c -lm || return 1
    "$tmp/spad_summary" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/permtest && cargo run --release --quiet -- "$root" ); }

run "SQL, the synthetic summaries"     sqlite3 check_sql
run "C, the real ablation table"       cc      check_c
run "Go, structure and every summary"  go      check_go
run "R, inference on the pin effect"   Rscript Rscript verify/verify.R "$root"
run "Rust, exact permutation tests"    cargo   check_rust
run "JavaScript, the documents"        node    node verify/docnumbers.js "$root"
run "Ruby, cross file identities"      ruby    ruby verify/crosscheck.rb "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
