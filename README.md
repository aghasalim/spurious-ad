# SpuriousAD — a planted-confound benchmark that refutes its own premise

[![ci](https://github.com/aghasalim/spurious-ad/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/spurious-ad/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A controlled anomaly-detection dataset where every anomalous image contains a
**true defect** and a spatially-disjoint **spurious mark** correlated with the
label, plus a metric for whether a detector's heatmap lands on the defect or the
mark. Built by a third-year Applied Computer Science (AI) student.

I built it to show that an unsupervised detector can score a perfect AUROC while
pointing at the wrong region. **It doesn't — and finding out why is the result.**

---

## The headline

Sweeping confound–label correlation ρ from 0 to 1, three seeds, PatchCore-style
detector trained on normal images only (`make sweep`):

| ρ | image AUROC | CAR | CAR random control | peak on defect |
|---|---|---|---|---|
| 0.00 | **1.000** | 0.133 | 0.621 | 93.2% |
| 0.25 | **1.000** | 0.143 | 0.599 | 91.1% |
| 0.50 | **1.000** | 0.144 | 0.588 | 97.4% |
| 0.75 | **1.000** | 0.187 | 0.607 | 93.9% |
| 1.00 | **1.000** | **0.560** | 0.610 | 80.8% |

**AUROC is 1.000 at every level.** The metric the whole field reports is
perfectly blind to whether a spurious mark is driving the score — that part of
the hypothesis holds completely.

CAR (Confound Attribution Ratio — share of heat on the mark rather than the
defect) rises 4.2×. That looks like the predicted collapse. It isn't:

- CAR at ρ=1 is **0.560 against a random-heatmap control of 0.610**. The detector
  never becomes confound-*seeking*; it decays to roughly what uniform noise would
  produce, which is set by the two regions' relative areas.
- The hottest single pixel still lands on the real defect **80.8%** of the time.

## Then the ablation killed the premise outright

Raising ρ does two things at once, and they are not the same thing:

- **A — label shortcut.** The detector exploits the mark *because it predicts the
  label*. This is the Clever Hans story.
- **B — training-set absence.** Raising ρ drives P(mark | normal) toward 0, so the
  mark stops appearing in the normal-only training set and becomes genuinely
  out-of-distribution.

Pinning the training rate at 0.465 while varying ρ separates them (`make mechanism`):

| training confound rate | ρ=0 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|
| free (falls to **0.000** at ρ=1) | 0.133 | 0.144 | **0.560** |
| **pinned at 0.465** | 0.133 | 0.117 | **0.130** |

**With the training rate held fixed, CAR does not move — even at ρ=1.0, where the
mark predicts the label perfectly.** Label correlation contributes essentially
nothing. Every bit of the apparent collapse was mechanism B.

In hindsight this is obvious, which is the useful part: **an unsupervised detector
never sees a label, so a label shortcut is not mechanically available to it.**
PatchCore models P(normal) and flags departures from it. A mark that is absent
from training and present at test time genuinely *is* a departure — flagging it is
arguably correct behaviour, not a Clever Hans effect.

So the naive way to build this benchmark — plant a label-correlated confound and
measure localisation — measures the wrong thing. It produces a real, reproducible,
4.2× effect that has nothing to do with the phenomenon it claims to study.

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

## The metric

```
CAR = mass(confound) / (mass(defect) + mass(confound))
```

over a normalised heatmap, on anomalous images that carry a confound. 0 = all
evidence on the real defect, 1 = all on the spurious mark.

Three things it does deliberately:

- **Mass, not peak.** A peak-based score is decided by one pixel and is unstable
  across seeds. `peak_on_defect` is reported separately because it is the
  operator-facing question — "the tool pointed here; is the defect there?"
- **Restricted to two regions.** Absolute heat varies with contrast and detector
  calibration. A ratio between the two regions that matter is comparable across
  detectors, at the cost of ignoring diffuse background — which
  `background_share` reports rather than hides (it is ~0.89 throughout, so most
  heat is in neither region, and that is worth knowing).
- **Always against a random control.** CAR has a nonzero null (~0.61 here) set by
  the regions' relative areas. Reading CAR without it would have made 0.560 look
  like catastrophic failure instead of near-chance.

---

## Running it

```bash
make setup && make test
```

8 tests, all on the generator and metric — the instrument, not the model.

```bash
make sweep && make mechanism
```

CPU or Apple MPS, a few minutes each. No dataset download: the data is
generated, which is what makes exact two-region ground truth possible.

---

## Honest limitations

- **Synthetic only.** Textures are sinusoidal, defects are blobs. The mechanism
  finding does not depend on realism — it is about what the training distribution
  contains — but the specific CAR values would move on MVTec.
- **One detector family.** PatchCore-style memory-bank kNN. PaDiM and reverse
  distillation share the "model normal, flag departures" structure, so I expect
  the argument to carry, but I have not run them.
- **Random subsampling, not greedy coreset selection.** Greedy k-center is
  PatchCore's efficiency contribution and does not change what the memory
  represents.
- **The negative result is the contribution.** There is no demonstration here of
  a genuine unsupervised Clever Hans effect — only a demonstration that this
  construction does not produce one.

## License

MIT — see [LICENSE](LICENSE).
