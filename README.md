# SpuriousAD, a planted-confound benchmark that refutes its own premise

> The synthetic finding now **replicates on real MVTec images** across two
> detector families and two backbones, see [External validity](#5-external-validity).
> Pinning the training-set confound rate halves CAR at perfect label
> correlation, so the effect is train-set absence, not a label shortcut.

[![ci](https://github.com/aghasalim/spurious-ad/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/spurious-ad/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A controlled anomaly-detection dataset where every anomalous image contains a
**true defect** and a spatially-disjoint **spurious mark** correlated with the
label, plus a metric for whether a detector's heatmap lands on the defect or the
mark. Built by a third-year Applied Computer Science (AI) student.

I built it to show that an unsupervised detector can score a perfect AUROC while
pointing at the wrong region. **It doesn't, and finding out why is the result.**

---


---

## Abstract

A natural way to build a controlled Clever Hans benchmark for unsupervised
anomaly detection is to plant an artefact that correlates with the defect label
and then measure whether the detector's heatmap moves onto it. This work builds
that benchmark, reproduces a large effect with it, and then shows the effect is
not the phenomenon it appears to be.

As the planted confound strengthens, image AUROC rises to 0.993 (PaDiM) and 0.998
(PatchCore) while the fraction of heatmap peaks landing on the real defect falls
from about 0.75 to 0.49 and 0.37. Reproducible, and present across categories and
across a backbone swap.

The ablation refutes the interpretation. Raising the confound strength does two
things at once: it makes the artefact predict the label, and it drives the
artefact out of the normal-only training set. Pinning the training confound rate
at 0.465 separates them, and with it pinned, the collapse disappears entirely,
even at the strength where the artefact predicts the label perfectly. In
hindsight this is mechanical: an unsupervised detector never sees a label, so a
label shortcut is not available to it. What it reacts to is a distributional
departure, and flagging that is arguably correct behaviour.

The contribution is therefore negative and methodological. The intuitive
construction produces a real, reproducible, 4.2x effect that has nothing to do
with the phenomenon it claims to study.

**Contributions.** (i) A planted-confound benchmark with a confound attribution
ratio and a random-attribution baseline. (ii) An ablation pinning the training
rate, which removes the effect. (iii) External validity checks across MVTec
categories, two detectors and two backbones. (iv) A negative result about how not
to construct this benchmark.

---

## 1. The headline

Sweeping confound, label correlation ρ from 0 to 1, three seeds, PatchCore-style
detector trained on normal images only (`make sweep`):

| ρ | image AUROC | CAR | CAR random control | peak on defect |
|---|---|---|---|---|
| 0.00 | **1.000** | 0.133 | 0.621 | 93.2% |
| 0.25 | **1.000** | 0.143 | 0.599 | 91.1% |
| 0.50 | **1.000** | 0.144 | 0.588 | 97.4% |
| 0.75 | **1.000** | 0.187 | 0.607 | 93.9% |
| 1.00 | **1.000** | **0.560** | 0.610 | 80.8% |

**AUROC is 1.000 at every level.** The metric the whole field reports is
perfectly blind to whether a spurious mark is driving the score, that part of
the hypothesis holds completely.

CAR (Confound Attribution Ratio, share of heat on the mark rather than the
defect) rises 4.2×. That looks like the predicted collapse. It isn't:

- CAR at ρ=1 is **0.560 against a random-heatmap control of 0.610**. The detector
  never becomes confound-*seeking*; it decays to roughly what uniform noise would
  produce, which is set by the two regions' relative areas.
- The hottest single pixel still lands on the real defect **80.8%** of the time.

![AUROC rises while localisation collapses](reports/figures/dissociation.png)

![the confound alone reaches AUROC 1.000](reports/figures/confound-alone.png)

## 2. Then the ablation killed the premise outright
Raising rho does two things at once: it makes the mark predict the label, and it
drives the mark out of the normal-only training set. Pinning the training
confound rate at 0.465 holds the second one still while rho varies. With it
pinned, CAR at rho=1 is 0.130, against 0.560 when the rate is left free and 0.133
at rho=0. Label correlation contributes nothing, so the whole 4.2x effect was the
mark going missing from training.

![pinning the training rate removes the collapse](reports/figures/mechanism.png)

![the same ablation walked from rho 0 to rho 1](reports/figures/rho-sweep.gif)

*Each detector walks from rho 0 to rho 1 through AUROC against peak on defect. Solid arms let the mark fall out of the normal-only training set, dashed arms pin that rate, and only the solid ones collapse.*

Full detail in [notes/METHODS.md](notes/METHODS.md#2-then-the-ablation-killed-the-premise-outright).
### What this does and does not say about prior work

It does not contradict [Kauffmann et al., *The Clever Hans Effect in Unsupervised
Learning*](https://arxiv.org/abs/2408.08041). They audit models post-hoc on native
data and find Clever Hans effects arising from **inductive bias in the learning
machinery**. That is a different mechanism from label correlation, and nothing here
tests it. What this repo shows is that the intuitive way to *construct* a
controlled version of their setting does not reproduce the phenomenon, because it
smuggles in a distributional artefact instead.

It is also not a criticism of [AUPIMO](https://arxiv.org/abs/2401.01984), whose
X-axis already penalises false positives on normal images. CAR differs in having a
notion of a **specific competing region**, which is what makes "the heat went
*there* instead" measurable at all.

---

## 3. The metric
``` CAR = mass(confound) / (mass(defect) + mass(confound)) ``` over a normalised heatmap, on anomalous images that carry a confound.

0 means all the evidence sits on the real defect, 1 means all of it sits on the
spurious mark. The null is not 0. A random heatmap scores about 0.61 here,
because the two regions' relative areas set the floor, so every CAR is read
against that control. Most of the heat lands in neither region, background share
runs 0.84 to 0.90 across the sweep, and the metric reports that instead of hiding
it.

![attribution against the random-attribution baseline](reports/figures/attribution.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#3-the-metric).
## 4. Running it

```bash
make setup && make test
```

17 tests, on the generator, the metric and the MVTec loader, the
instrument, not the model.

```bash
make sweep && make mechanism
```

CPU or Apple MPS, a few minutes each. No dataset download: the data is
generated, which is what makes exact two-region ground truth possible.

---

## 5. External validity
The synthetic result above says label correlation does not drive the confound attribution, the CAR rise is the mark going *out of distribution* in the normal-only training set, not the detector learning a label shortcut it cannot mechanically learn.
That is a claim about mechanism, so it has to hold on real images. `make
real-mechanism` plants the same confound in five MVTec AD categories and repeats
the pin there. At rho=1.0 the pin takes PatchCore CAR from 0.415 to 0.189 and
PaDiM from 0.387 to 0.201 against a null of 0.430, and PatchCore's hottest pixel
lands on the real defect 74.8% of the time instead of 36.9%. A resnet18 sweep
with no pin backs this from the other side: CAR climbs 0.21 to 0.39 across rho
and never reaches its 0.43 null.

![per-category localisation](reports/figures/by-category.png)
![the same sweep under a different backbone](reports/figures/backbone.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#5-external-validity).
## 6. Limitations

- **Synthetic textures are the core; MVTec is the external check.** The mechanism
  finding now holds on real images (above), but the headline synthetic CAR values
  are specific to the generator, the *ordering and the mechanism* are what
  transfer, not the exact numbers.
- **The resnet18 arm is a sweep, not the pinned ablation.** It corroborates but
  does not independently establish the mechanism claim.
- **Random subsampling, not greedy coreset selection.** Greedy k-center is
  PatchCore's efficiency contribution and does not change what the memory
  represents.
- **The negative result is the contribution.** There is no demonstration here of
  a genuine unsupervised Clever Hans effect, only a demonstration that this
  construction does not produce one.

## 7. Everything here is recomputed in seven languages

Every number in this README, in [notes/METHODS.md](notes/METHODS.md) and in
every figure comes out of one pandas groupby, in
[`experiments/sweep.py`](experiments/sweep.py),
[`experiments/mechanism.py`](experiments/mechanism.py) and
[`experiments/real.py`](experiments/real.py). The tests check the generator and
the metric, which is the instrument. Nothing checked the tables, and nothing
downstream could have, because the figures read the same CSV the README quotes.

So the eight tables in `reports/` are recomputed from the raw per-run JSON next
to them by seven implementations in seven other languages, and CI fails if any
two disagree. A mistake would have to be repeated identically in all of them to
survive.

| implementation | what it recomputes | measured agreement |
| --- | --- | --- |
| [`verify/summaries.sql`](verify/summaries.sql) | the two synthetic summaries, as a group by in SQLite over `json_each` | 59 cells, largest gap 4.97e-05 |
| [`verify/summary.c`](verify/summary.c) | the real ablation table: means, the sample sd of CAR, the n sum, columns resolved by name | 108 cells, largest gap 4.97e-05 |
| [`verify/gocheck`](verify/gocheck) | all eight tables, plus the structure of all 13 files in `reports/` | 867 cells, largest gap 4.97e-05 |
| [`verify/verify.R`](verify/verify.R) | the cell means, then a paired test and a category cluster interval on the pin effect | 36 cells, largest gap 4.57e-05 |
| [`verify/permtest`](verify/permtest) | the same two questions by exhaustive enumeration, 543,038 statistics | exact, no sampling error |
| [`verify/docnumbers.js`](verify/docnumbers.js) | the 74 figures quoted in the two documents, against the cells they were copied from | all 74 agree |
| [`verify/crosscheck.rb`](verify/crosscheck.rb) | cell balance, the two groupbys folding into each other, reruns of a shared configuration | 513 runs, 171 cells; shared runs identical at 0.0e+00 |

The published tables are rounded to four decimals, so an exact recomputation can
differ from them by at most 5e-05. Every implementation is held to 6e-05, and
the largest gap any of them found anywhere is 4.97e-05, which is the rounding
and nothing else.

Run them all with [`./verify/verify.sh`](verify/verify.sh). Each is skipped with
a message if its toolchain is missing, so a partial install still runs the rest.

**The ablation now has a test behind it.** The pin was reported as a difference
of two means and left there. The design is paired, one run at pin off for every
run at pin on with the same category, seed and detector, so R takes the 15
paired differences at rho=1 and Rust walks all 2^15 sign assignments rather than
assuming they are normal. PatchCore: mean +0.2254, t 4.47, exact p 4.883e-04.
PaDiM: mean +0.1859, t 4.82, exact p 4.883e-04. That p is the smallest this data
can produce, because three of the fifteen pairs differ by exactly zero.
Resampling whole categories rather than runs, which is the honest unit when five
MVTec categories are the population of interest, puts the PatchCore effect at
[0.0803, 0.4014] and PaDiM at [0.0714, 0.3274]. R samples that interval 20,000
times, Rust enumerates all 5^5 = 3,125 of them, and the two agree to four
decimals on all four endpoints.

**With a control.** At rho=0 the natural training confound rate already equals
the pinned rate, so pinning is a no-op there and every paired difference is
exactly zero, exact p 1.000. A test that cannot come back null is not a test.

**One thing the checks changed.** The resnet18 sweep is described as staying
below its random-heatmap null at every correlation. It does, as an ordering of
the published means, and Go, R and Rust all reproduce those means. Run as a
paired test on the individual runs, the separation is 4.8 to 10.6 standard
errors at rho of 0.75 and below, and 1.1 for PatchCore and 1.5 for PaDiM at
rho=1. The ordering at rho=1 is a point estimate, not a separation this data can
defend, so the harness requires only the ordering there. Writing these checks
also caught the test count in section 4, which still said 8 after the MVTec
tests were added.

**The harness is itself checked.** CI corrupts
`reports/real_mechanism_summary.csv`, requires the harness to reject it,
restores it and requires a pass. Each implementation catches what it is
responsible for and nothing more:

| what was corrupted | caught by |
| --- | --- |
| a published CAR in `sweep_summary.csv`, 0.5601 to 0.5701 | SQL, Go, JavaScript |
| a published CAR in `real_mechanism_summary.csv`, 0.4146 to 0.4246 | C, Go, R, Rust, JavaScript, Ruby |
| one run's CAR in `real_mechanism.json` | C, Go, R, Rust, Ruby |
| a cell replaced by `nan` in `real_sweep_summary.csv` | Go, Ruby |
| an extra field on one row of `real_sweep_by_category.csv` | Go |
| one run deleted from `real_backbone.json` | Go, Ruby |
| the headline effect written as 5.2x in this README | JavaScript |
| the pinned CAR values rotated between categories, leaving every per-detector cell mean untouched | Go, R, Rust |

## 8. Licence


MIT, see [LICENSE](LICENSE).

## References

The papers and sources this implementation follows. Each one is here because
the code uses the method, the dataset or the metric it describes.

- **Roth, Pemula, Zepeda, Schölkopf, Brox, Gehler. Towards Total Recall in Industrial Anomaly Detection. CVPR 2022.** [arXiv:2106.08265](https://arxiv.org/abs/2106.08265) PatchCore.
- **Defard, Setkov, Loesch, Audigier. PaDiM: a Patch Distribution Modeling Framework. ICPR 2021.** [arXiv:2011.08785](https://arxiv.org/abs/2011.08785) PaDiM.
- **Bergmann, Fauser, Sattlegger, Steger. MVTec AD. CVPR 2019.** the real image dataset the planted confound is replicated on.
- **Geirhos, Jacobsen, Michaelis et al. Shortcut Learning in Deep Neural Networks. Nature Machine Intelligence 2, 2020.** [arXiv:2004.07780](https://arxiv.org/abs/2004.07780) the failure mode this benchmark was built to plant deliberately.
