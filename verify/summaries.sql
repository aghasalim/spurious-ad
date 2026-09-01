-- Recompute reports/sweep_summary.csv and reports/mechanism_summary.csv from
-- the raw per-run rows in reports/sweep.json and reports/mechanism.json.
--
-- Those two tables are the synthetic core of the README: the headline table in
-- section 1 and the ablation in section 2. Both are a pandas groupby in
-- experiments/sweep.py and experiments/mechanism.py, and every figure drawn
-- from them reads the same output, so an error in the aggregation would not
-- show up anywhere downstream.
--
-- A groupby is what SQL is for, so this redoes it in SQLite, joins the result
-- to the published CSV, and prints a line per cell. Any line beginning FAIL is
-- a disagreement; verify/verify.sh treats one as an error.
--
-- Run from the repository root:
--   sqlite3 -init verify/summaries.sql :memory: ""

.bail on
.mode list
.separator ' '
.headers off

.import --csv reports/sweep_summary.csv sweep_pub
.import --csv reports/mechanism_summary.csv mech_pub

-- Half of the last published decimal place is 5e-5. The rest is headroom for
-- summation order; a change of a whole unit in the last place still fails.
CREATE TEMP VIEW tol AS SELECT 6e-5 AS t;

CREATE TEMP VIEW sweep_runs AS
SELECT json_extract(value, '$.rho')                  AS rho,
       json_extract(value, '$.auroc')                AS auroc,
       json_extract(value, '$.confound_alone_auroc') AS confound_auroc,
       json_extract(value, '$.car')                  AS car,
       json_extract(value, '$.car_random')           AS car_random,
       json_extract(value, '$.peak_on_defect')       AS peak_on_defect,
       json_extract(value, '$.background_share')     AS background
FROM json_each(readfile('reports/sweep.json'));

CREATE TEMP VIEW mech_runs AS
SELECT json_extract(value, '$.rho')                  AS rho,
       json_extract(value, '$.pinned_train_rate')    AS pinned,
       json_extract(value, '$.train_confound_rate')  AS train_conf_rate,
       json_extract(value, '$.auroc')                AS auroc,
       json_extract(value, '$.car')                  AS car,
       json_extract(value, '$.peak_on_defect')       AS peak_on_defect
FROM json_each(readfile('reports/mechanism.json'));

CREATE TEMP VIEW sweep_agg AS
SELECT rho, count(*) AS k,
       avg(auroc) AS auroc, avg(confound_auroc) AS confound_auroc,
       avg(car) AS car, avg(car_random) AS car_random,
       avg(peak_on_defect) AS peak_on_defect, avg(background) AS background
FROM sweep_runs GROUP BY rho;

-- Sample standard deviation, ddof = 1, which is what pandas std() returns.
-- Two pass against the group mean rather than sum of squares minus the square
-- of the sum, which loses digits when the spread is small next to the mean.
CREATE TEMP VIEW sweep_sd AS
SELECT r.rho AS rho,
       sqrt(sum((r.car - a.car) * (r.car - a.car)) / (a.k - 1)) AS car_sd
FROM sweep_runs r JOIN sweep_agg a ON a.rho = r.rho
GROUP BY r.rho;

CREATE TEMP VIEW mech_agg AS
SELECT rho, pinned, count(*) AS k,
       avg(train_conf_rate) AS train_conf_rate, avg(auroc) AS auroc,
       avg(car) AS car, avg(peak_on_defect) AS peak_on_defect
FROM mech_runs GROUP BY rho, pinned;

CREATE TEMP VIEW cells AS
    SELECT 'sweep_summary.csv' AS f, 'rho=' || a.rho AS k, 'auroc' AS m,
           a.auroc AS got, CAST(p.auroc AS REAL) AS want
      FROM sweep_agg a JOIN sweep_pub p ON CAST(p.rho AS REAL) = a.rho
    UNION ALL SELECT 'sweep_summary.csv', 'rho=' || a.rho, 'confound_auroc',
           a.confound_auroc, CAST(p.confound_auroc AS REAL)
      FROM sweep_agg a JOIN sweep_pub p ON CAST(p.rho AS REAL) = a.rho
    UNION ALL SELECT 'sweep_summary.csv', 'rho=' || a.rho, 'car',
           a.car, CAST(p.car AS REAL)
      FROM sweep_agg a JOIN sweep_pub p ON CAST(p.rho AS REAL) = a.rho
    UNION ALL SELECT 'sweep_summary.csv', 'rho=' || s.rho, 'car_sd',
           s.car_sd, CAST(p.car_sd AS REAL)
      FROM sweep_sd s JOIN sweep_pub p ON CAST(p.rho AS REAL) = s.rho
    UNION ALL SELECT 'sweep_summary.csv', 'rho=' || a.rho, 'car_random',
           a.car_random, CAST(p.car_random AS REAL)
      FROM sweep_agg a JOIN sweep_pub p ON CAST(p.rho AS REAL) = a.rho
    UNION ALL SELECT 'sweep_summary.csv', 'rho=' || a.rho, 'peak_on_defect',
           a.peak_on_defect, CAST(p.peak_on_defect AS REAL)
      FROM sweep_agg a JOIN sweep_pub p ON CAST(p.rho AS REAL) = a.rho
    UNION ALL SELECT 'sweep_summary.csv', 'rho=' || a.rho, 'background',
           a.background, CAST(p.background AS REAL)
      FROM sweep_agg a JOIN sweep_pub p ON CAST(p.rho AS REAL) = a.rho

    UNION ALL SELECT 'mechanism_summary.csv',
           'pinned=' || a.pinned || ',rho=' || a.rho, 'train_conf_rate',
           a.train_conf_rate, CAST(p.train_conf_rate AS REAL)
      FROM mech_agg a JOIN mech_pub p
        ON CAST(p.rho AS REAL) = a.rho
       AND (CASE p.pinned_train_rate WHEN 'True' THEN 1 ELSE 0 END) = a.pinned
    UNION ALL SELECT 'mechanism_summary.csv',
           'pinned=' || a.pinned || ',rho=' || a.rho, 'auroc',
           a.auroc, CAST(p.auroc AS REAL)
      FROM mech_agg a JOIN mech_pub p
        ON CAST(p.rho AS REAL) = a.rho
       AND (CASE p.pinned_train_rate WHEN 'True' THEN 1 ELSE 0 END) = a.pinned
    UNION ALL SELECT 'mechanism_summary.csv',
           'pinned=' || a.pinned || ',rho=' || a.rho, 'car',
           a.car, CAST(p.car AS REAL)
      FROM mech_agg a JOIN mech_pub p
        ON CAST(p.rho AS REAL) = a.rho
       AND (CASE p.pinned_train_rate WHEN 'True' THEN 1 ELSE 0 END) = a.pinned
    UNION ALL SELECT 'mechanism_summary.csv',
           'pinned=' || a.pinned || ',rho=' || a.rho, 'peak_on_defect',
           a.peak_on_defect, CAST(p.peak_on_defect AS REAL)
      FROM mech_agg a JOIN mech_pub p
        ON CAST(p.rho AS REAL) = a.rho
       AND (CASE p.pinned_train_rate WHEN 'True' THEN 1 ELSE 0 END) = a.pinned;

-- A published row that has no matching cell in the raw runs, or the other way
-- round, would drop silently out of the join above, so count both sides.
SELECT 'FAIL sweep_summary.csv has ' || (SELECT count(*) FROM sweep_pub)
       || ' rows but sweep.json forms ' || (SELECT count(*) FROM sweep_agg) || ' cells'
WHERE (SELECT count(*) FROM sweep_pub) <> (SELECT count(*) FROM sweep_agg);

SELECT 'FAIL mechanism_summary.csv has ' || (SELECT count(*) FROM mech_pub)
       || ' rows but mechanism.json forms ' || (SELECT count(*) FROM mech_agg) || ' cells'
WHERE (SELECT count(*) FROM mech_pub) <> (SELECT count(*) FROM mech_agg);

SELECT 'FAIL only ' || count(*) || ' cells joined, expected 59' FROM cells
HAVING count(*) <> 59;

SELECT 'FAIL ' || f || ' ' || k || ' ' || m
       || ' recomputed ' || printf('%.10f', got)
       || ' published ' || printf('%.4f', want)
       || ' |d| ' || printf('%.2e', abs(got - want))
FROM cells, tol WHERE abs(got - want) > t;

SELECT '  ' || f || '  ' || count(*) || ' cells  max |d| ' || printf('%.2e', max(abs(got - want)))
FROM cells GROUP BY f;

SELECT 'SQL reproduces ' || count(*) || ' cells of the two synthetic summaries from '
       || 'the raw runs, max |d| ' || printf('%.2e', max(abs(got - want)))
FROM cells;
