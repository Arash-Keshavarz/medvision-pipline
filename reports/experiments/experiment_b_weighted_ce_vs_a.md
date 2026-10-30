# Experiment B vs. A: class-weighted vs. plain cross-entropy (ResNet18)

**Single-seed comparison on one validation split.** Differences below are
observations from one run each, not evidence that either loss is better.
The test split has not been loaded or evaluated. Validation results only;
not a clinical performance estimate.

**Decision:** experiment A is the *provisional* reference baseline because
it has the higher validation macro F1, the selection metric chosen before
these runs. This is a selection rule, not a finding of statistical
superiority. Experiment B is preserved and its recall/precision trade-offs
are documented below.

## Setup

| | A (plain CE) | B (class-weighted CE) |
|---|---|---|
| Run ID | `baseline-20261030-130625` | `baseline-20261030-172401` |
| Run directory (ignored) | `runs/baseline_resnet18/…` | `runs/resnet18_weighted_ce/…` |
| Commit (worktree clean) | `72c7bbe` | `639a338` |
| Config | `configs/baseline_resnet18.yaml` | `configs/resnet18_weighted_ce.yaml` |
| Elapsed / epochs | 3,192 s / 14 (early stop) | 3,370 s / 13 (early stop) |
| Best epoch (max val macro F1) | 9 | 8 |

Everything else was identical, verified from both `resolved_config.json`
files (only `loss` and `output_root` differ; the `data` sections are equal):

- train and validation manifests (SHA-256 `e87d97e4…` and `37709067…`);
- seed 42 and loader seed 42, preprocessing, and batch size 16;
- AdamW with lr 1e-4 and weight decay 1e-4;
- 20 maximum epochs, patience 5, and selection by validation macro F1;
- ImageNet initialisation (B did not start from A's checkpoint).

B's class weights come from training-manifest counts,
`weight[c] = N_train / (7 * count[c])`, and are saved in `class_weights.json`:

| | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| Train count | 230 | 361 | 770 | 80 | 778 | 4695 | 99 |
| Weight | 4.356 | 2.775 | 1.301 | 12.523 | 1.288 | 0.213 | 10.120 |

## Best-checkpoint validation metrics

| Metric | A (epoch 9) | B (epoch 8) | B − A |
|---|---|---|---|
| Macro F1 | **0.7025** | 0.6854 | −0.017 |
| Balanced accuracy | 0.6908 | **0.7150** | +0.024 |
| Accuracy | **0.8208** | 0.7668 | −0.054 |
| Macro precision | 0.7236 | 0.6800 | −0.044 |
| Validation unweighted CE | **0.6017** | 0.6589 | +0.057 |
| Train unweighted CE (augmented, during epoch) | 0.2490 | 0.4824 | |

The unweighted cross-entropy is the per-sample mean computed the same way
for both runs. B's weighted validation loss (1.0533 at epoch 8) is a
different quantity and is not compared with A.

## Melanoma

| | A | B |
|---|---|---|
| Precision | 0.525 | 0.467 |
| Recall | 0.569 | 0.629 |
| F1 | 0.546 | 0.536 |
| mel → nv errors (of 167 melanomas) | 65 (38.9%) | 29 (17.4%) |
| nv → mel errors (of 1,005 naevi) | 52 (5.2%) | 98 (9.8%) |
| mel → bkl errors | 6 | 32 |
| Images predicted as mel | 181 | 225 |

B moved melanoma errors rather than reducing them. Fewer melanomas were
called `nv`, but more were called `bkl`, and twice as many naevi were called
`mel`. Melanoma recall rose by 10 images (95 to 105), and its F1 was about
unchanged.

## Per-class metrics (best checkpoints)

| Class | Support | A P | A R | A F1 | B P | B R | B F1 |
|---|---|---|---|---|---|---|---|
| akiec | 49 | 0.791 | 0.694 | 0.739 | 0.769 | 0.612 | 0.682 |
| bcc | 77 | 0.753 | 0.714 | 0.733 | 0.806 | 0.701 | 0.750 |
| bkl | 165 | 0.804 | 0.545 | 0.650 | 0.484 | 0.836 | 0.613 |
| df | 17 | 0.450 | 0.529 | 0.486 | 0.529 | 0.529 | 0.529 |
| mel | 167 | 0.525 | 0.569 | 0.546 | 0.467 | 0.629 | 0.536 |
| nv | 1005 | 0.886 | 0.926 | 0.906 | 0.944 | 0.792 | 0.861 |
| vasc | 21 | 0.857 | 0.857 | 0.857 | 0.760 | 0.905 | 0.826 |

## Confusion matrices (rows true, columns predicted)

A, epoch 9:

| true \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| akiec | 34 | 2 | 5 | 2 | 0 | 6 | 0 |
| bcc | 4 | 55 | 1 | 4 | 3 | 10 | 0 |
| bkl | 4 | 6 | 90 | 3 | 26 | 34 | 2 |
| df | 0 | 0 | 0 | 9 | 5 | 3 | 0 |
| mel | 0 | 1 | 6 | 0 | 95 | 65 | 0 |
| nv | 1 | 8 | 10 | 2 | 52 | 931 | 1 |
| vasc | 0 | 1 | 0 | 0 | 0 | 2 | 18 |

B, epoch 8:

| true \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| akiec | 30 | 1 | 10 | 0 | 7 | 1 | 0 |
| bcc | 5 | 54 | 11 | 1 | 3 | 2 | 1 |
| bkl | 2 | 2 | 138 | 1 | 9 | 12 | 1 |
| df | 0 | 0 | 4 | 9 | 3 | 1 | 0 |
| mel | 0 | 0 | 32 | 0 | 105 | 29 | 1 |
| nv | 2 | 10 | 90 | 6 | 98 | 796 | 3 |
| vasc | 0 | 0 | 0 | 0 | 0 | 2 | 19 |

Under B, `bkl` became the main sink for misclassifications: 90 naevi and
32 melanomas were predicted as `bkl`. Its recall rose from 0.545 to 0.836,
and its precision fell from 0.804 to 0.484. `nv` predictions fell from 70.0%
to 56.2% of images, against a true share of 67.0%.

## Learning curves and stability

B, per epoch (weighted CE is B's training objective; unweighted CE is
comparable with A's losses):

| Epoch | Train wCE | Train uCE | Val wCE | Val uCE | Val acc | Val bal. acc | Val macro F1 |
|---|---|---|---|---|---|---|---|
| 1 | 1.147 | 1.012 | 0.855 | 0.934 | 0.648 | 0.687 | 0.474 |
| 2 | 0.767 | 0.763 | 0.851 | 0.742 | 0.715 | 0.701 | 0.580 |
| 3 | 0.672 | 0.679 | 0.898 | 0.737 | 0.710 | 0.717 | 0.623 |
| 4 | 0.559 | 0.608 | 1.053 | 0.883 | 0.672 | 0.651 | 0.527 |
| 5 | 0.492 | 0.565 | 0.849 | 0.655 | 0.770 | 0.717 | 0.637 |
| 6 | 0.489 | 0.563 | 0.845 | 0.601 | 0.782 | 0.713 | 0.644 |
| 7 | 0.431 | 0.505 | 0.763 | 0.629 | 0.761 | 0.732 | 0.647 |
| **8** | 0.396 | 0.482 | 1.053 | 0.659 | 0.767 | 0.715 | **0.685** |
| 9 | 0.394 | 0.457 | 1.240 | 0.802 | 0.736 | 0.662 | 0.550 |
| 10 | 0.436 | 0.505 | 0.731 | 0.611 | 0.763 | 0.741 | 0.672 |
| 11 | 0.374 | 0.451 | 0.997 | 0.612 | 0.760 | 0.682 | 0.618 |
| 12 | 0.330 | 0.415 | 0.949 | 0.673 | 0.766 | 0.692 | 0.638 |
| 13 | 0.313 | 0.397 | 1.239 | 0.752 | 0.765 | 0.668 | 0.610 |

A's per-epoch curves are in `experiment_a_baseline_resnet18.md`.

Validation macro F1 stability over all epochs:

| | A | B |
|---|---|---|
| Std. deviation across epochs | 0.023 | 0.060 |
| Mean abs. change between epochs | 0.019 | 0.064 |
| Largest change between epochs | 0.042 | 0.136 (epoch 8 to 9) |
| Lowest val unweighted CE (epoch) | 0.536 (4) | 0.601 (6) |
| Highest val balanced accuracy (epoch) | 0.691 (9) | 0.741 (10) |

These figures describe how each run's validation metrics moved *within*
its own training trajectory. They are not estimates of between-seed
uncertainty, which cannot be measured from one run per configuration.

- **B's trajectory is much less smooth.** Validation macro F1 moves by up to
  0.14, and weighted validation loss by up to 0.5, between neighbouring
  epochs; train weighted CE rose at epoch 10.
- *Hypothesis (untested):* the large weights on `df` (12.5) and `vasc`
  (10.1), with batch size 16, may make each batch's gradient depend strongly
  on whether a rare-class image is present. No experiment here isolates this.
- **Training fit.** B's train accuracy at its best epoch was 0.81 against
  0.91 for A. This alone does not show that B overfits less. Training metrics
  are measured on augmented images during the epoch, and B optimises a
  different objective that down-weights `nv`, which dominates accuracy. A's
  validation unweighted CE rises from epoch 4 while its training loss keeps
  falling, which is consistent with overfitting in A. B's validation curves
  are too irregular to read a comparable trend from.
- **B's selected epoch is an isolated peak.** Epoch 8 sits between 0.647
  and 0.550. Balanced accuracy peaked at a different epoch (10, 0.741). B's
  best-checkpoint figures therefore depend strongly on which epoch was
  selected.

## Summary

On this seed and split, class weighting traded accuracy and precision for
recall on some classes. Balanced accuracy rose by 0.024, while macro F1 fell
by 0.017 and accuracy by 0.054. Melanoma-to-naevus errors fell from 65 to 29,
but naevus-to-melanoma errors nearly doubled (52 to 98), and many errors
shifted to `bkl`. Melanoma F1 was essentially unchanged. B's learning curves
are markedly less smooth than A's.

A is provisionally selected as the reference baseline under the predefined
macro-F1 rule. With one seed per configuration, the between-seed variance is
unknown, so it cannot be said whether the gaps of 0.017 in macro F1 or 0.024
in balanced accuracy exceed run-to-run variation. Repeated runs with
different seeds would be needed to estimate that.

## Limitations

- One seed per configuration; no confidence intervals or significance tests.
- Best epochs were selected on the same validation split that is reported,
  so both best-checkpoint results are optimistically biased. Selecting the
  maximum of a more irregular curve may bias B's figure further, but this was
  not quantified.
- `df` (17) and `vasc` (21) have very small validation support.
- Per-class metrics are saved for best epochs only.
- B's run directory is named `baseline-…` because the run-name prefix is
  the run type (`baseline` vs. `smoke`), not the experiment. This is cosmetic;
  the loss is recorded in `run_metadata.json` and `resolved_config.json`.
