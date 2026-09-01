/* Recompute reports/real_mechanism_summary.csv from reports/real_mechanism.json.
 *
 * That one table carries the external-validity claim in section 5 of the
 * README: the pin takes PatchCore CAR from 0.415 to 0.189 and PaDiM from 0.387
 * to 0.201. It is a pandas groupby over 180 runs, written by
 * experiments/real.py, and the figures beside it are drawn from its own output,
 * so nothing downstream could have caught a mistake in it.
 *
 * This is the tight kernel: the group means, the sample standard deviation of
 * CAR across runs, and the integer sum of n, in C, from the raw runs. Columns
 * of the published CSV are resolved by name, so a column inserted upstream
 * cannot silently shift what is compared.
 *
 * The JSON reader assumes the two-space pretty-printed layout json.dumps
 * produces, one scalar per line. That is what experiments/real.py writes, and a
 * change to it fails loudly here rather than quietly.
 *
 * Run: cc -std=c99 -O2 -o summary verify/summary.c -lm && ./summary <repo root>
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE 4096
#define MAX_RUNS 1024
#define MAX_GROUPS 64
#define NAME 64

/* The published CSVs are rounded to four decimals, so an exact recomputation
 * can differ by at most half of the last place. The headroom above that is for
 * summation order; a change of one unit in the last published place is still
 * rejected. */
#define TOL 6e-5

typedef struct {
    char detector[NAME];
    int pinned;
    double rho;
    double auroc, confound_auroc, train_rate, car, car_random, peak, background;
    long n;
} Run;

typedef struct {
    char detector[NAME];
    int pinned;
    double rho;
    Run *runs[MAX_RUNS];
    int count;
} Group;

/* --- a reader for the flat pretty-printed JSON experiments/real.py writes --- */

static int json_key(const char *line, char *key, size_t cap, const char **value)
{
    const char *a = strchr(line, '"');
    if (!a) return 0;
    const char *b = strchr(a + 1, '"');
    if (!b) return 0;
    size_t n = (size_t)(b - a - 1);
    if (n >= cap) n = cap - 1;
    memcpy(key, a + 1, n);
    key[n] = '\0';

    const char *c = strchr(b + 1, ':');
    if (!c) return 0;
    c++;
    while (*c == ' ') c++;
    *value = c;
    return 1;
}

static void json_string(const char *value, char *out, size_t cap)
{
    const char *a = strchr(value, '"');
    if (!a) { out[0] = '\0'; return; }
    const char *b = strchr(a + 1, '"');
    if (!b) { out[0] = '\0'; return; }
    size_t n = (size_t)(b - a - 1);
    if (n >= cap) n = cap - 1;
    memcpy(out, a + 1, n);
    out[n] = '\0';
}

static int load_runs(const char *path, Run *runs, int cap)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }

    char line[LINE], key[NAME];
    const char *value;
    int count = -1, fields = 0;

    while (fgets(line, sizeof line, f)) {
        const char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '{') {
            if (++count >= cap) { fprintf(stderr, "too many runs\n"); fclose(f); return -1; }
            memset(&runs[count], 0, sizeof runs[count]);
            fields = 0;
            continue;
        }
        if (count < 0 || !json_key(line, key, sizeof key, &value)) continue;

        Run *r = &runs[count];
        if      (!strcmp(key, "detector"))           json_string(value, r->detector, NAME);
        else if (!strcmp(key, "pinned_train_rate"))  r->pinned = (*value == 't');
        else if (!strcmp(key, "rho"))                r->rho = atof(value);
        else if (!strcmp(key, "auroc"))              r->auroc = atof(value);
        else if (!strcmp(key, "confound_alone_auroc")) r->confound_auroc = atof(value);
        else if (!strcmp(key, "train_confound_rate")) r->train_rate = atof(value);
        else if (!strcmp(key, "car"))                r->car = atof(value);
        else if (!strcmp(key, "car_random"))         r->car_random = atof(value);
        else if (!strcmp(key, "peak_on_defect"))     r->peak = atof(value);
        else if (!strcmp(key, "background_share"))   r->background = atof(value);
        else if (!strcmp(key, "n"))                  r->n = atol(value);
        else continue;
        fields++;
    }
    fclose(f);
    if (count < 0) { fprintf(stderr, "%s holds no records\n", path); return -1; }
    (void)fields;
    return count + 1;
}

/* --- CSV, columns resolved by name --- */

static int column_of(const char *header, const char *name)
{
    char buf[LINE];
    strncpy(buf, header, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';
    int i = 0;
    for (char *tok = strtok(buf, ",\r\n"); tok; tok = strtok(NULL, ",\r\n"), i++)
        if (!strcmp(tok, name)) return i;
    return -1;
}

static const char *field(const char *line, int index)
{
    static char out[256];
    int col = 0;
    const char *p = line;
    while (col < index) {
        p = strchr(p, ',');
        if (!p) return NULL;
        p++; col++;
    }
    const char *end = strchr(p, ',');
    size_t n = end ? (size_t)(end - p) : strlen(p);
    if (n >= sizeof out) n = sizeof out - 1;
    memcpy(out, p, n);
    out[n] = '\0';
    char *nl = strpbrk(out, "\r\n");
    if (nl) *nl = '\0';
    return out;
}

/* --- the aggregations experiments/real.py performs --- */

static double mean_of(Group *g, size_t offset)
{
    double s = 0.0;
    for (int i = 0; i < g->count; i++)
        s += *(double *)((char *)g->runs[i] + offset);
    return s / g->count;
}

/* Sample standard deviation, ddof = 1, which is what pandas std() returns.
 * Two pass, so the cancellation the one pass form suffers cannot bite. */
static double car_sd_of(Group *g)
{
    if (g->count < 2) return NAN;
    const double mu = mean_of(g, offsetof(Run, car));
    double ss = 0.0;
    for (int i = 0; i < g->count; i++) {
        const double d = g->runs[i]->car - mu;
        ss += d * d;
    }
    return sqrt(ss / (g->count - 1));
}

static long n_of(Group *g)
{
    long s = 0;
    for (int i = 0; i < g->count; i++) s += g->runs[i]->n;
    return s;
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024], line[LINE], header[LINE];
    static Run runs[MAX_RUNS];
    static Group groups[MAX_GROUPS];
    int n_groups = 0;

    snprintf(path, sizeof path, "%s/reports/real_mechanism.json", root);
    const int n_runs = load_runs(path, runs, MAX_RUNS);
    if (n_runs < 0) return 2;

    for (int i = 0; i < n_runs; i++) {
        Run *r = &runs[i];
        int g = -1;
        for (int k = 0; k < n_groups; k++)
            if (!strcmp(groups[k].detector, r->detector) && groups[k].pinned == r->pinned
                && groups[k].rho == r->rho)
                g = k;
        if (g < 0) {
            if (n_groups >= MAX_GROUPS) { fprintf(stderr, "too many groups\n"); return 2; }
            g = n_groups++;
            strncpy(groups[g].detector, r->detector, NAME - 1);
            groups[g].pinned = r->pinned;
            groups[g].rho = r->rho;
            groups[g].count = 0;
        }
        groups[g].runs[groups[g].count++] = r;
    }
    printf("read %d runs from real_mechanism.json in %d cells\n", n_runs, n_groups);

    snprintf(path, sizeof path, "%s/reports/real_mechanism_summary.csv", root);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!fgets(header, sizeof header, f)) { fclose(f); return 2; }

    static const char *cols[] = { "detector", "pinned_train_rate", "rho", "auroc",
        "confound_auroc", "train_conf_rate", "car", "car_sd", "car_random",
        "peak_on_defect", "background", "n" };
    int idx[12];
    for (int c = 0; c < 12; c++) {
        idx[c] = column_of(header, cols[c]);
        if (idx[c] < 0) {
            fprintf(stderr, "real_mechanism_summary.csv has no %s column\n", cols[c]);
            fclose(f);
            return 2;
        }
    }

    int rows = 0, failures = 0, cells = 0;
    double worst = 0.0;

    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;

        char det[NAME];
        strncpy(det, field(line, idx[0]), NAME - 1);
        det[NAME - 1] = '\0';
        const int pinned = !strcmp(field(line, idx[1]), "True");
        const double rho = atof(field(line, idx[2]));

        Group *g = NULL;
        for (int k = 0; k < n_groups; k++)
            if (!strcmp(groups[k].detector, det) && groups[k].pinned == pinned
                && groups[k].rho == rho)
                g = &groups[k];
        if (!g) {
            printf("  %-10s pin=%d rho=%.2f  no such cell in the raw runs  FAIL\n",
                   det, pinned, rho);
            failures++;
            rows++;
            continue;
        }

        const double got[9] = {
            mean_of(g, offsetof(Run, auroc)),
            mean_of(g, offsetof(Run, confound_auroc)),
            mean_of(g, offsetof(Run, train_rate)),
            mean_of(g, offsetof(Run, car)),
            car_sd_of(g),
            mean_of(g, offsetof(Run, car_random)),
            mean_of(g, offsetof(Run, peak)),
            mean_of(g, offsetof(Run, background)),
            (double)n_of(g),
        };

        double row_worst = 0.0;
        int bad = 0;
        for (int c = 0; c < 9; c++) {
            const double want = atof(field(line, idx[c + 3]));
            const double d = fabs(got[c] - want);
            cells++;
            if (d > row_worst) row_worst = d;
            if (d > TOL) {
                bad++;
                printf("      %s: recomputed %.10f, published %.4f, |d| %.2e\n",
                       cols[c + 3], got[c], want, d);
            }
        }
        if (row_worst > worst) worst = row_worst;
        printf("  %-10s pin=%-5s rho=%.2f  %2d runs  car %.6f  sd %.6f  n %5ld  "
               "max |d| %.2e  %s\n",
               det, pinned ? "True" : "False", rho, g->count, got[3], got[4],
               (long)got[8], row_worst, bad ? "FAIL" : "ok");
        failures += bad;
        rows++;
    }
    fclose(f);

    if (rows != n_groups) {
        fprintf(stderr, "\nreal_mechanism_summary.csv has %d rows but the raw runs "
                        "form %d cells\n", rows, n_groups);
        return 1;
    }
    if (failures) {
        printf("\n%d cells disagree with reports/real_mechanism_summary.csv\n", failures);
        return 1;
    }
    printf("\nC reproduces all %d cells of real_mechanism_summary.csv from the raw runs,\n"
           "max |d| %.2e against a tolerance of %.0e\n", cells, worst, TOL);
    return 0;
}
