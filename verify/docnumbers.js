// Does the prose still say what the data says?
//
// Every number in README.md and notes/METHODS.md was copied out of a table in
// reports/ by hand. The tables get regenerated; the prose does not. Nothing
// anywhere connected the two, so a rerun that moved a value would leave the
// documents quietly wrong, and a reader has no way to tell.
//
// This pins each quoted figure to the cell it came from. A figure is accepted
// when it equals the published cell rounded to the number of decimals the
// document prints, which is the rule a person is following when they write
// 0.560 for 0.5601. Claims hedged with "about" or "~" are deliberately left
// alone: they are not assertions of a value.
//
// Run: node verify/docnumbers.js <repo root>

'use strict';
const fs = require('fs');
const path = require('path');

const root = process.argv[2] || '.';
let failures = 0;
let figures = 0;

function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}

function readCsv(name) {
  const lines = read(path.join('reports', name)).trim().split('\n');
  const head = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const cells = line.split(',');
    const row = {};
    head.forEach((h, i) => { row[h] = cells[i]; });
    return row;
  });
}

function readJson(name) {
  return JSON.parse(read(path.join('reports', name)));
}

// Selectors are matched numerically where the value looks like a number, so
// "1.0" in the CSV and 1 in the selector are the same rho.
function cell(name, where, column) {
  const rows = readCsv(name).filter((r) =>
    Object.keys(where).every((k) => {
      const want = where[k];
      return typeof want === 'number' ? Number(r[k]) === want : r[k] === String(want);
    }));
  if (rows.length !== 1) {
    throw new Error(`${name}: ${JSON.stringify(where)} selects ${rows.length} rows`);
  }
  return Number(rows[0][column]);
}

function column(name, where, col) {
  return readCsv(name).filter((r) =>
    Object.keys(where).every((k) => {
      const want = where[k];
      return typeof want === 'number' ? Number(r[k]) === want : r[k] === String(want);
    })).map((r) => Number(r[col]));
}

function decimals(text) {
  const dot = text.indexOf('.');
  return dot < 0 ? 0 : text.length - dot - 1;
}

function agrees(quoted, actual) {
  const d = decimals(quoted);
  const factor = Math.pow(10, d);
  return Number(quoted) === Math.round(actual * factor) / factor;
}

const docs = {};
for (const name of ['README.md', 'notes/METHODS.md']) {
  // Dropping blockquote markers and collapsing whitespace lets a claim be
  // checked without caring where the paragraph happens to wrap.
  docs[name] = read(name).replace(/^\s*>\s?/gm, '').replace(/\s+/g, ' ');
}

function claim(doc, label, re, wanted) {
  const hits = docs[doc].match(new RegExp(re.source, re.flags + 'g')) || [];
  if (hits.length !== 1) {
    console.log(`  FAIL ${doc}: "${label}" matches ${hits.length} places, expected 1`);
    failures++;
    return;
  }
  const m = docs[doc].match(re);
  const got = m.slice(1, wanted.length + 1);
  figures += got.length;
  const bad = [];
  got.forEach((q, i) => {
    if (!agrees(q, wanted[i].value)) {
      bad.push(`${wanted[i].from} is ${wanted[i].value}, the text says ${q}`);
    }
  });
  if (bad.length) {
    console.log(`  FAIL ${doc}: ${label}`);
    bad.forEach((b) => console.log(`         ${b}`));
    failures += bad.length;
  } else {
    console.log(`  ok   ${doc.padEnd(16)} ${label}  (${got.join(', ')})`);
  }
}

const of = (value, from) => ({ value, from });

// --- the headline table in section 1, row by row -----------------------------

console.log('the headline table in README section 1, against reports/sweep_summary.csv');
{
  const wanted = ['auroc', 'car', 'car_random', 'peak_on_defect'];
  const rowRe = /^\| (\d\.\d\d) \| \*\*([\d.]+)\*\* \| \*?\*?([\d.]+)\*?\*? \| ([\d.]+) \| ([\d.]+)% \|$/;
  const rows = read('README.md').split('\n').map((l) => l.trim().match(rowRe)).filter(Boolean);
  if (rows.length !== 5) {
    console.log(`  FAIL the table has ${rows.length} rows, expected 5`);
    failures++;
  }
  for (const m of rows) {
    const rho = Number(m[1]);
    const quoted = [m[2], m[3], m[4], m[5]];
    let bad = 0;
    wanted.forEach((col, i) => {
      const actual = cell('sweep_summary.csv', { rho }, col);
      const scaled = col === 'peak_on_defect' ? actual * 100 : actual;
      if (!agrees(quoted[i], scaled)) {
        console.log(`  FAIL rho=${rho} ${col}: table says ${quoted[i]}, ` +
                    `sweep_summary.csv says ${actual}`);
        bad++;
      }
    });
    failures += bad;
    figures += quoted.length;
    if (!bad) console.log(`  ok   rho=${m[1]}  ${quoted.join('  ')}`);
  }
}

// --- claims in the prose -----------------------------------------------------

console.log('\nthe numbers quoted in the prose');

const S = (rho, col) => cell('sweep_summary.csv', { rho }, col);
const M = (pin, rho, col) =>
  cell('mechanism_summary.csv', { pinned_train_rate: pin, rho }, col);
const RM = (det, pin, rho, col) =>
  cell('real_mechanism_summary.csv', { detector: det, pinned_train_rate: pin, rho }, col);
const RB = (det, rho, col) => cell('real_backbone_summary.csv', { detector: det, rho }, col);

claim('README.md', 'abstract, AUROC at rho=1 on real images',
  /image AUROC rises to ([\d.]+) \(PaDiM\) and ([\d.]+) \(PatchCore\)/,
  [of(RM('padim', 'False', 1, 'auroc'), 'real_mechanism_summary padim rho=1 auroc'),
   of(RM('patchcore', 'False', 1, 'auroc'), 'real_mechanism_summary patchcore rho=1 auroc')]);

claim('README.md', 'abstract, peak on defect at rho=1',
  /to ([\d.]+) and ([\d.]+)\. Reproducible/,
  [of(RM('padim', 'False', 1, 'peak_on_defect'), 'real_mechanism_summary padim rho=1 peak'),
   of(RM('patchcore', 'False', 1, 'peak_on_defect'), 'real_mechanism_summary patchcore rho=1 peak')]);

claim('README.md', 'abstract, the pinned training rate',
  /Pinning the training confound rate at ([\d.]+) separates/,
  [of(M('True', 1, 'train_conf_rate'), 'mechanism_summary pinned train_conf_rate')]);

claim('README.md', 'abstract, the size of the effect',
  /reproducible, ([\d.]+)x effect/,
  [of(S(1, 'car') / S(0, 'car'), 'sweep_summary car at rho=1 over rho=0')]);

claim('README.md', 'section 1, the rise in CAR',
  /rises ([\d.]+)\u00d7/,
  [of(S(1, 'car') / S(0, 'car'), 'sweep_summary car at rho=1 over rho=0')]);

claim('README.md', 'section 1, CAR against its null',
  /CAR at .=1 is \*\*([\d.]+) against a random-heatmap control of ([\d.]+)\*\*/,
  [of(S(1, 'car'), 'sweep_summary car at rho=1'),
   of(S(1, 'car_random'), 'sweep_summary car_random at rho=1')]);

claim('README.md', 'section 1, the hottest pixel at rho=1',
  /hottest single pixel still lands on the real defect \*\*([\d.]+)%\*\*/,
  [of(S(1, 'peak_on_defect') * 100, 'sweep_summary peak_on_defect at rho=1')]);

claim('README.md', 'section 2, the pinned rate',
  /Pinning the training confound rate at ([\d.]+) holds/,
  [of(M('True', 1, 'train_conf_rate'), 'mechanism_summary pinned train_conf_rate')]);

claim('README.md', 'section 2, the ablation',
  /With it pinned, CAR at rho=1 is ([\d.]+), against ([\d.]+) when the rate is left free and ([\d.]+) at rho=0/,
  [of(M('True', 1, 'car'), 'mechanism_summary pinned rho=1 car'),
   of(M('False', 1, 'car'), 'mechanism_summary free rho=1 car'),
   of(M('False', 0, 'car'), 'mechanism_summary free rho=0 car')]);

claim('README.md', 'section 2, the effect the pin removes',
  /the whole ([\d.]+)x effect was the mark going missing/,
  [of(S(1, 'car') / S(0, 'car'), 'sweep_summary car at rho=1 over rho=0')]);

claim('README.md', 'section 3, the background share range',
  /background share runs ([\d.]+) to ([\d.]+) across the sweep/,
  [of(Math.min(...column('sweep_summary.csv', {}, 'background')), 'sweep_summary lowest background'),
   of(Math.max(...column('sweep_summary.csv', {}, 'background')), 'sweep_summary highest background')]);

claim('README.md', 'section 5, the pin on real images',
  /the pin takes PatchCore CAR from ([\d.]+) to ([\d.]+) and PaDiM from ([\d.]+) to ([\d.]+) against a null of ([\d.]+)/,
  [of(RM('patchcore', 'False', 1, 'car'), 'real_mechanism_summary patchcore free rho=1 car'),
   of(RM('patchcore', 'True', 1, 'car'), 'real_mechanism_summary patchcore pinned rho=1 car'),
   of(RM('padim', 'False', 1, 'car'), 'real_mechanism_summary padim free rho=1 car'),
   of(RM('padim', 'True', 1, 'car'), 'real_mechanism_summary padim pinned rho=1 car'),
   of(RM('patchcore', 'False', 1, 'car_random'), 'real_mechanism_summary rho=1 car_random')]);

claim('README.md', 'section 5, the hottest pixel under the pin',
  /lands on the real defect ([\d.]+)% of the time instead of ([\d.]+)%/,
  [of(RM('patchcore', 'True', 1, 'peak_on_defect') * 100, 'real_mechanism_summary patchcore pinned rho=1 peak'),
   of(RM('patchcore', 'False', 1, 'peak_on_defect') * 100, 'real_mechanism_summary patchcore free rho=1 peak')]);

claim('README.md', 'section 5, the resnet18 sweep',
  /CAR climbs ([\d.]+) to ([\d.]+) across rho and never reaches its ([\d.]+) null/,
  [of(RB('patchcore', 0, 'car'), 'real_backbone_summary patchcore rho=0 car'),
   of(RB('patchcore', 1, 'car'), 'real_backbone_summary patchcore rho=1 car'),
   of(Math.max(...column('real_backbone_summary.csv', {}, 'car_random')), 'real_backbone_summary highest car_random')]);

claim('notes/METHODS.md', 'the free arm of the ablation table',
  /\| free \(falls to \*\*([\d.]+)\*\* at .=1\) \| ([\d.]+) \| ([\d.]+) \| \*\*([\d.]+)\*\* \|/,
  [of(M('False', 1, 'train_conf_rate'), 'mechanism_summary free rho=1 train_conf_rate'),
   of(M('False', 0, 'car'), 'mechanism_summary free rho=0 car'),
   of(M('False', 0.5, 'car'), 'mechanism_summary free rho=0.5 car'),
   of(M('False', 1, 'car'), 'mechanism_summary free rho=1 car')]);

claim('notes/METHODS.md', 'the pinned arm of the ablation table',
  /\| \*\*pinned at ([\d.]+)\*\* \| ([\d.]+) \| ([\d.]+) \| \*\*([\d.]+)\*\* \|/,
  [of(M('True', 1, 'train_conf_rate'), 'mechanism_summary pinned train_conf_rate'),
   of(M('True', 0, 'car'), 'mechanism_summary pinned rho=0 car'),
   of(M('True', 0.5, 'car'), 'mechanism_summary pinned rho=0.5 car'),
   of(M('True', 1, 'car'), 'mechanism_summary pinned rho=1 car')]);

claim('notes/METHODS.md', 'the PatchCore rows of the real ablation table',
  /\| PatchCore \| free . \*\*([\d.]+)\*\* \| \*\*([\d.]+)\*\* \| ([\d.]+) \| ([\d.]+)% \| \| PatchCore \| \*\*pinned at ([\d.]+)\*\* \| \*\*([\d.]+)\*\* \| ([\d.]+) \| \*\*([\d.]+)%\*\* \|/,
  [of(RM('patchcore', 'False', 1, 'train_conf_rate'), 'real_mechanism_summary patchcore free rho=1 train_conf_rate'),
   of(RM('patchcore', 'False', 1, 'car'), 'real_mechanism_summary patchcore free rho=1 car'),
   of(RM('patchcore', 'False', 1, 'car_random'), 'real_mechanism_summary patchcore free rho=1 car_random'),
   of(RM('patchcore', 'False', 1, 'peak_on_defect') * 100, 'real_mechanism_summary patchcore free rho=1 peak'),
   of(RM('patchcore', 'True', 1, 'train_conf_rate'), 'real_mechanism_summary patchcore pinned rho=1 train_conf_rate'),
   of(RM('patchcore', 'True', 1, 'car'), 'real_mechanism_summary patchcore pinned rho=1 car'),
   of(RM('patchcore', 'True', 1, 'car_random'), 'real_mechanism_summary patchcore pinned rho=1 car_random'),
   of(RM('patchcore', 'True', 1, 'peak_on_defect') * 100, 'real_mechanism_summary patchcore pinned rho=1 peak')]);

claim('notes/METHODS.md', 'the PaDiM rows of the real ablation table',
  /\| PaDiM \| free . \*\*([\d.]+)\*\* \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)% \| \| PaDiM \| \*\*pinned at ([\d.]+)\*\* \| \*\*([\d.]+)\*\* \| ([\d.]+) \| \*\*([\d.]+)%\*\* \|/,
  [of(RM('padim', 'False', 1, 'train_conf_rate'), 'real_mechanism_summary padim free rho=1 train_conf_rate'),
   of(RM('padim', 'False', 1, 'car'), 'real_mechanism_summary padim free rho=1 car'),
   of(RM('padim', 'False', 1, 'car_random'), 'real_mechanism_summary padim free rho=1 car_random'),
   of(RM('padim', 'False', 1, 'peak_on_defect') * 100, 'real_mechanism_summary padim free rho=1 peak'),
   of(RM('padim', 'True', 1, 'train_conf_rate'), 'real_mechanism_summary padim pinned rho=1 train_conf_rate'),
   of(RM('padim', 'True', 1, 'car'), 'real_mechanism_summary padim pinned rho=1 car'),
   of(RM('padim', 'True', 1, 'car_random'), 'real_mechanism_summary padim pinned rho=1 car_random'),
   of(RM('padim', 'True', 1, 'peak_on_defect') * 100, 'real_mechanism_summary padim pinned rho=1 peak')]);

claim('notes/METHODS.md', 'the resnet18 sweep against its null',
  /it rises ([\d.]+) . ([\d.]+) as . goes 0 . 1 while the null sits at ([\d.]+)/,
  [of(RB('patchcore', 0, 'car'), 'real_backbone_summary patchcore rho=0 car'),
   of(RB('patchcore', 1, 'car'), 'real_backbone_summary patchcore rho=1 car'),
   of(Math.max(...column('real_backbone_summary.csv', {}, 'car_random')), 'real_backbone_summary highest car_random')]);

// --- claims about the shape of the experiment, checked against the runs ------

console.log('\nthe grid the documents describe, against the raw runs');
{
  const runs = readJson('real_mechanism.json');
  const uniq = (f) => [...new Set(runs.map((r) => r[f]))];
  const facts = [
    ['five MVTec AD categories', /five MVTec AD categories/, uniq('category').length, 5],
    ['two detector families', /two detector families/, uniq('detector').length, 2],
    ['three seeds', /three seeds/, uniq('seed').length, 3],
  ];
  for (const [label, re, got, want] of facts) {
    const said = re.test(docs['README.md']);
    if (!said) {
      console.log(`  FAIL README.md no longer says "${label}"`);
      failures++;
    } else if (got !== want) {
      console.log(`  FAIL README.md says "${label}" but real_mechanism.json has ${got}`);
      failures++;
    } else {
      console.log(`  ok   README.md says "${label}", the runs have ${got}`);
    }
  }

  // The "never reaches its null" claim is an inequality, not a value.
  const car = column('real_backbone_summary.csv', {}, 'car');
  const nul = column('real_backbone_summary.csv', {}, 'car_random');
  if (Math.max(...car) >= Math.min(...nul)) {
    console.log(`  FAIL the resnet18 CAR reaches its null: max CAR ${Math.max(...car)}, ` +
                `lowest null ${Math.min(...nul)}`);
    failures++;
  } else {
    console.log(`  ok   resnet18 CAR stays below its null: highest CAR ${Math.max(...car)}, ` +
                `lowest null ${Math.min(...nul)}`);
  }
}

console.log();
if (failures > 0) {
  console.log(`${failures} figures in the documents no longer match reports/`);
  process.exit(1);
}
console.log(`all ${figures} figures quoted in README.md and notes/METHODS.md still match`);
console.log('the cell in reports/ each was taken from');
