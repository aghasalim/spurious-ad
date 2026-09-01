// Structural validation of everything under reports/, plus an independent
// recomputation of all eight published summary tables from the raw per-run
// JSON they were aggregated from.
//
// Every CSV in reports/ is a pandas groupby of one of the JSON files next to
// it, written by experiments/sweep.py, experiments/mechanism.py and
// experiments/real.py. Nothing checked that those aggregations are right, and
// nothing checked that the files are well formed: a truncated write, a column
// that drifted, a NaN escaping a division on an empty cell would all sit there
// unnoticed until a reader trusted the table.
//
// This walks every file and then redoes every groupby from the raw rows.
//
// Run: cd verify/gocheck && go run . -root ../..
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// The published CSVs are rounded to four decimals, so an exact recomputation
// can differ from them by at most half of the last place, 5e-5. The extra
// 1e-5 is headroom for summation order across languages; anything that moves a
// published cell by a whole unit in the last place is still rejected.
const tol = 6e-5

type metric struct{ col, field, op string } // op is mean, sd or sum

type spec struct {
	src, pub string
	keys     []string
	metrics  []metric
}

// mean of the run level column, except car_sd which is the sample standard
// deviation across runs in the cell, and n which is a sum. These mirror the
// .agg() calls in experiments/*.py exactly, which is the point: if one of them
// is wrong, this disagrees.
var runMetrics = []metric{
	{"auroc", "auroc", "mean"},
	{"confound_auroc", "confound_alone_auroc", "mean"},
	{"train_conf_rate", "train_confound_rate", "mean"},
	{"car", "car", "mean"},
	{"car_sd", "car", "sd"},
	{"car_random", "car_random", "mean"},
	{"peak_on_defect", "peak_on_defect", "mean"},
	{"background", "background_share", "mean"},
	{"n", "n", "sum"},
}

var byCategoryMetrics = []metric{
	{"auroc", "auroc", "mean"},
	{"car", "car", "mean"},
	{"car_random", "car_random", "mean"},
	{"peak_on_defect", "peak_on_defect", "mean"},
}

var specs = []spec{
	{"sweep.json", "sweep_summary.csv", []string{"rho"}, []metric{
		{"auroc", "auroc", "mean"},
		{"confound_auroc", "confound_alone_auroc", "mean"},
		{"car", "car", "mean"},
		{"car_sd", "car", "sd"},
		{"car_random", "car_random", "mean"},
		{"peak_on_defect", "peak_on_defect", "mean"},
		{"background", "background_share", "mean"},
	}},
	{"mechanism.json", "mechanism_summary.csv", []string{"pinned_train_rate", "rho"}, []metric{
		{"train_conf_rate", "train_confound_rate", "mean"},
		{"auroc", "auroc", "mean"},
		{"car", "car", "mean"},
		{"peak_on_defect", "peak_on_defect", "mean"},
	}},
	{"real_sweep.json", "real_sweep_summary.csv", []string{"detector", "rho"}, runMetrics},
	{"real_backbone.json", "real_backbone_summary.csv", []string{"detector", "rho"}, runMetrics},
	{"real_mechanism.json", "real_mechanism_summary.csv",
		[]string{"detector", "pinned_train_rate", "rho"}, runMetrics},
	{"real_sweep.json", "real_sweep_by_category.csv",
		[]string{"category", "detector", "rho"}, byCategoryMetrics},
	{"real_mechanism.json", "real_mechanism_by_category.csv",
		[]string{"category", "detector", "rho"}, byCategoryMetrics},
	{"real_backbone.json", "real_backbone_by_category.csv",
		[]string{"category", "detector", "rho"}, byCategoryMetrics},
}

func readJSON(path string) ([]map[string]any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rows []map[string]any
	// Rejects NaN and Infinity outright: they are not JSON, and json.dumps
	// emits them, so a run that produced one would land here as a parse error.
	if err := json.Unmarshal(b, &rows); err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("no records")
	}
	return rows, nil
}

func readCSV(path string) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("only %d rows", len(rows))
	}
	return rows[0], rows[1:], nil
}

// keyText renders a group key value the same way whether it came from JSON or
// from the CSV, so the two sides can be matched without depending on how
// either printed its floats.
func keyText(v any) string {
	switch t := v.(type) {
	case bool:
		if t {
			return "True"
		}
		return "False"
	case float64:
		return strconv.FormatFloat(t, 'g', -1, 64)
	case string:
		return t
	}
	return fmt.Sprint(v)
}

func csvKeyText(cell string) string {
	if cell == "True" || cell == "False" {
		return cell
	}
	if f, err := strconv.ParseFloat(cell, 64); err == nil {
		return strconv.FormatFloat(f, 'g', -1, 64)
	}
	return cell
}

func num(v any) (float64, bool) {
	f, ok := v.(float64)
	if !ok || math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, false
	}
	return f, true
}

func aggregate(vals []float64, op string) float64 {
	n := float64(len(vals))
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	switch op {
	case "sum":
		return sum
	case "mean":
		return sum / n
	case "sd": // sample standard deviation, pandas std(), ddof=1
		if len(vals) < 2 {
			return math.NaN()
		}
		mu := sum / n
		ss := 0.0
		for _, v := range vals {
			ss += (v - mu) * (v - mu)
		}
		return math.Sqrt(ss / (n - 1))
	}
	return math.NaN()
}

// validateCSV reports every structural problem in one file rather than the
// first, so a broken run is diagnosed in a single pass.
func validateCSV(path string) []string {
	var problems []string
	header, rows, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}
	seen := map[string]bool{}
	for _, h := range header {
		if strings.TrimSpace(h) == "" {
			problems = append(problems, "a column has an empty name")
		}
		if seen[h] {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}
	for i, row := range rows {
		for j, cell := range row {
			low := strings.ToLower(strings.TrimSpace(cell))
			if low == "" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is empty", i+2, header[j]))
			}
			if low == "nan" || low == "inf" || low == "-inf" || low == "infinity" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is %s", i+2, header[j], cell))
			}
		}
	}
	return problems
}

// validateJSON insists every record carries the identical key set. A run that
// crashed halfway and wrote a short record would otherwise average silently
// over whichever rows happened to have the column.
func validateJSON(path string) []string {
	rows, err := readJSON(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}
	var first []string
	for k := range rows[0] {
		first = append(first, k)
	}
	sort.Strings(first)
	want := strings.Join(first, ",")

	var problems []string
	for i, r := range rows {
		var keys []string
		for k, v := range r {
			keys = append(keys, k)
			if v == nil {
				problems = append(problems,
					fmt.Sprintf("record %d has a null %s", i, k))
			}
			if f, ok := v.(float64); ok && (math.IsNaN(f) || math.IsInf(f, 0)) {
				problems = append(problems,
					fmt.Sprintf("record %d has a non-finite %s", i, k))
			}
		}
		sort.Strings(keys)
		if got := strings.Join(keys, ","); got != want {
			problems = append(problems,
				fmt.Sprintf("record %d has a different key set to record 0", i))
		}
	}
	return problems
}

func recompute(reports string, s spec) (int, float64, []string) {
	rows, err := readJSON(filepath.Join(reports, s.src))
	if err != nil {
		return 0, 0, []string{fmt.Sprintf("%s: %v", s.src, err)}
	}
	header, pub, err := readCSV(filepath.Join(reports, s.pub))
	if err != nil {
		return 0, 0, []string{fmt.Sprintf("%s: %v", s.pub, err)}
	}

	// Columns are resolved by name, so a column added or reordered upstream
	// cannot silently shift what is compared.
	colOf := map[string]int{}
	for i, h := range header {
		colOf[h] = i
	}
	for _, k := range s.keys {
		if _, ok := colOf[k]; !ok {
			return 0, 0, []string{fmt.Sprintf("%s has no %s column", s.pub, k)}
		}
	}

	groups := map[string]map[string][]float64{}
	for i, r := range rows {
		var parts []string
		for _, k := range s.keys {
			v, ok := r[k]
			if !ok {
				return 0, 0, []string{fmt.Sprintf("%s record %d has no %s", s.src, i, k)}
			}
			parts = append(parts, keyText(v))
		}
		key := strings.Join(parts, "|")
		if groups[key] == nil {
			groups[key] = map[string][]float64{}
		}
		for _, m := range s.metrics {
			f, ok := num(r[m.field])
			if !ok {
				return 0, 0, []string{fmt.Sprintf("%s record %d: %s is not a finite number",
					s.src, i, m.field)}
			}
			groups[key][m.field+"/"+m.op] = append(groups[key][m.field+"/"+m.op], f)
		}
	}

	var problems []string
	if len(pub) != len(groups) {
		problems = append(problems, fmt.Sprintf(
			"%s has %d rows but the raw data has %d groups", s.pub, len(pub), len(groups)))
	}

	worst, checked := 0.0, 0
	for i, row := range pub {
		var parts []string
		for _, k := range s.keys {
			parts = append(parts, csvKeyText(row[colOf[k]]))
		}
		key := strings.Join(parts, "|")
		g, ok := groups[key]
		if !ok {
			problems = append(problems, fmt.Sprintf("%s row %d: group %s is not in %s",
				s.pub, i+2, key, s.src))
			continue
		}
		for _, m := range s.metrics {
			c, ok := colOf[m.col]
			if !ok {
				problems = append(problems, fmt.Sprintf("%s has no %s column", s.pub, m.col))
				continue
			}
			want, err := strconv.ParseFloat(strings.TrimSpace(row[c]), 64)
			if err != nil {
				problems = append(problems, fmt.Sprintf("%s row %d column %s is not a number: %q",
					s.pub, i+2, m.col, row[c]))
				continue
			}
			got := aggregate(g[m.field+"/"+m.op], m.op)
			d := math.Abs(got - want)
			checked++
			if d > worst {
				worst = d
			}
			if d > tol {
				problems = append(problems, fmt.Sprintf(
					"%s row %d %s: recomputed %.10f, published %.4f, |d| %.2e",
					s.pub, i+2, m.col, got, want, d))
			}
		}
	}
	return checked, worst, problems
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	reports := filepath.Join(*root, "reports")
	csvs, _ := filepath.Glob(filepath.Join(reports, "*.csv"))
	jsons, _ := filepath.Glob(filepath.Join(reports, "*.json"))
	sort.Strings(csvs)
	sort.Strings(jsons)
	if len(csvs) == 0 || len(jsons) == 0 {
		fmt.Fprintf(os.Stderr, "no data files under %s\n", reports)
		os.Exit(2)
	}

	bad := 0
	fmt.Printf("validating %d CSV and %d JSON files under reports/\n", len(csvs), len(jsons))
	for _, path := range append(append([]string{}, csvs...), jsons...) {
		var problems []string
		if strings.HasSuffix(path, ".csv") {
			problems = validateCSV(path)
		} else {
			problems = validateJSON(path)
		}
		for _, p := range problems {
			fmt.Printf("  %s: %s\n", filepath.Base(path), p)
		}
		bad += len(problems)
	}
	if bad == 0 {
		fmt.Printf("  no ragged rows, duplicate or empty columns, NaN, Inf, null " +
			"or short records\n")
	}

	fmt.Printf("\nrecomputing every published summary from the raw per-run rows\n")
	totalCells, totalWorst := 0, 0.0
	for _, s := range specs {
		checked, worst, problems := recompute(reports, s)
		status := "ok"
		if len(problems) > 0 {
			status = "FAIL"
			bad += len(problems)
		}
		fmt.Printf("  %-30s %-22s %3d cells  max |d| %.2e  %s\n",
			s.pub, "from "+s.src, checked, worst, status)
		for _, p := range problems {
			fmt.Printf("      %s\n", p)
		}
		totalCells += checked
		if worst > totalWorst {
			totalWorst = worst
		}
	}

	if bad > 0 {
		fmt.Printf("\n%d problems\n", bad)
		os.Exit(1)
	}
	fmt.Printf("\nGo reproduces all %d published cells from the raw runs, "+
		"max |d| %.2e against a tolerance of %.0e,\nand reports/ is well formed\n",
		totalCells, totalWorst, tol)
}
