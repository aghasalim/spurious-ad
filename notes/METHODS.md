# Methods and detail

Long form detail moved out of the README.


## 2. Then the ablation killed the premise outright


![pinning the training rate removes the collapse](reports/figures/mechanism.png)

Dashed lines are the pinned-training-rate control. The collapse disappears under
it, which is what rules out the label-shortcut reading.

Raising ρ does two things at once, and they are not the same thing:

- **A: label shortcut.** The detector exploits the mark *because it predicts the
  label*. This is the Clever Hans story.
- **B: training-set absence.** Raising ρ drives P(mark | normal) toward 0, so the
  mark stops appearing in the normal-only training set and becomes genuinely
  out-of-distribution.

Pinning the training rate at 0.465 while varying ρ separates them (`make mechanism`):

| training confound rate | ρ=0 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|
| free (falls to **0.000** at ρ=1) | 0.133 | 0.144 | **0.560** |
| **pinned at 0.465** | 0.133 | 0.117 | **0.130** |

**With the training rate held fixed, CAR does not move, even at ρ=1.0, where the
mark predicts the label perfectly.** Label correlation contributes essentially
nothing. Every bit of the apparent collapse was mechanism B.

In hindsight this is obvious, which is the useful part: **an unsupervised detector
never sees a label, so a label shortcut is not mechanically available to it.**
PatchCore models P(normal) and flags departures from it. A mark that is absent
from training and present at test time genuinely *is* a departure, flagging it is
arguably correct behaviour, not a Clever Hans effect.

So the naive way to build this benchmark, plant a label-correlated confound and
measure localisation, measures the wrong thing. It produces a real, reproducible,
4.2× effect that has nothing to do with the phenomenon it claims to study.


## 3. The metric


![attribution against the random-attribution baseline](reports/figures/attribution.png)

```
CAR = mass(confound) / (mass(defect) + mass(confound))
```

over a normalised heatmap, on anomalous images that carry a confound. 0 = all
evidence on the real defect, 1 = all on the spurious mark.

Three things it does deliberately:

- **Mass, not peak.** A peak-based score is decided by one pixel and is unstable
  across seeds.`peak_on_defect` is reported separately because it is the
  operator-facing question, "the tool pointed here; is the defect there?"
- **Restricted to two regions.** Absolute heat varies with contrast and detector
  calibration. A ratio between the two regions that matter is comparable across
  detectors, at the cost of ignoring diffuse background, which
`background_share` reports rather than hides (it is ~0.89 throughout, so most
  heat is in neither region, and that is worth knowing).
- **Always against a random control.** CAR has a nonzero null (~0.61 here) set by
  the regions' relative areas. Reading CAR without it would have made 0.560 look
  like catastrophic failure instead of near-chance.

---


## 5. External validity


![per-category localisation](reports/figures/by-category.png)

![the same sweep under a different backbone](reports/figures/backbone.png)

The synthetic result above says label correlation does not drive the confound
attribution, the CAR rise is the mark going *out of distribution* in the
normal-only training set, not the detector learning a label shortcut it cannot
mechanically learn. That is a claim about mechanism, so it has to survive real
images. It does.`make real-mechanism` runs the same planted confound on MVTec
AD, five categories (bottle, carpet, grid, hazelnut, tile), the identical
per-category random-heatmap null (which varies with region area, so it is
computed per category and never assumed).

The load-bearing test, at ρ=1.0 where the mark predicts the label perfectly
the only difference between the two rows is whether the mark stays in the
normal-only training set (`wide_resnet50_2`, 1,062 anomalous images/cell):

| detector | training rate | CAR | its null | peak on defect |
|---|---|---|---|---|
| PatchCore | free → **0.000** | **0.415** | 0.430 | 36.9% |
| PatchCore | **pinned at 0.51** | **0.189** | 0.430 | **74.8%** |
| PaDiM | free → **0.000** | 0.387 | 0.430 | 49.3% |
| PaDiM | **pinned at 0.51** | **0.201** | 0.430 | **74.9%** |

Pinning the training rate **halves CAR and doubles the peak-on-defect rate**, at
identical, perfect label correlation. With the pin in place CAR is flat across ρ.
The synthetic finding replicates on real images across **two detector families**: PatchCore (non-parametric kNN) and PaDiM (a fitted per-patch Gaussian), which
share only the "model normal, flag departures" structure the conclusion rests on.

A second backbone corroborates it. On`resnet18` (sweep only, no pin), CAR never
even reaches its null: it rises 0.21 → 0.39 as ρ goes 0 → 1 while the null sits
at 0.43, so the confound attribution stays *below chance* at every correlation
level. I did not run the pinned ablation on resnet18, the mechanism claim rests
on the`wide_resnet50_2` pin above; resnet18 only shows the effect is, if
anything, weaker on a smaller extractor.

One detail that protects the metric: the loader drops anomalous images whose
ground-truth mask touches the confound box (else the two regions overlap and CAR
is unscoreable) and masks that vanish at 256 px (a zero defect denominator would
score a spurious CAR of 1.0). Both counts are reported per result row, not
silently applied, across all 15 categories the constraint costs 0 to 8 images each.
