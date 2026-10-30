# Experiment A: ResNet18 baseline (plain cross-entropy)

Accepted reference run for later experiments. Validation results only; the
test split has not been loaded or evaluated. This is a research baseline,
not a clinical performance estimate.

## Provenance

| Item | Value |
|---|---|
| Run ID | `baseline-20261030-130625` |
| Run directory (ignored, not in Git) | `runs/baseline_resnet18/baseline-20261030-130625/` |
| Commit | `72c7bbe1ae85c1fb10afce44ea56373051c113e8` (worktree clean) |
| Config | `configs/baseline_resnet18.yaml` + `configs/data.yaml` |
| Train manifest SHA-256 | `e87d97e49deb84d1d2fc559cbe8247b832dc478881344626f190519ae1806d5e` (7,013 images) |
| Validation manifest SHA-256 | `377090670c858c18377f9d53491e79fa583d93999d79e744d5eb367e7e5a75d2` (1,501 images) |
| Environment | MPS, Python 3.11.9, torch 2.14.0, torchvision 0.29.0, seed 42 |
| Elapsed | 3,192 s (14 epochs, early stopped; patience 5) |

Settings: ImageNet-pretrained ResNet18 (`IMAGENET1K_V1`), full fine-tuning,
unweighted cross-entropy, AdamW (lr 1e-4, weight decay 1e-4), batch 16,
224 px training augmentation / resize-224 + centre-crop evaluation, maximum
20 epochs, checkpoint selection by validation macro F1.

Checkpoints (`best.pt`, `last.pt`, 134 MB each) stay in the ignored run
directory and are not tracked in Git.

## Results (validation split)

| | Best epoch (9, `best.pt`) | Last epoch (14, `last.pt`) |
|---|---|---|
| Macro F1 | **0.7025** | 0.6912 |
| Balanced accuracy | 0.6908 | 0.6694 |
| Accuracy | 0.8208 | 0.8108 |
| Validation loss (unweighted CE) | 0.6017 | 0.6812 |
| Train loss (unweighted CE) | 0.2490 | 0.1686 |

Best-epoch per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| akiec | 0.791 | 0.694 | 0.739 | 49 |
| bcc | 0.753 | 0.714 | 0.733 | 77 |
| bkl | 0.804 | 0.545 | 0.650 | 165 |
| df | 0.450 | 0.529 | 0.486 | 17 |
| mel | 0.525 | 0.569 | 0.546 | 167 |
| nv | 0.886 | 0.926 | 0.906 | 1005 |
| vasc | 0.857 | 0.857 | 0.857 | 21 |
| Macro | 0.724 | 0.691 | 0.703 | 1501 |

Best-epoch confusion matrix (rows true, columns predicted):

| true \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| akiec | 34 | 2 | 5 | 2 | 0 | 6 | 0 |
| bcc | 4 | 55 | 1 | 4 | 3 | 10 | 0 |
| bkl | 4 | 6 | 90 | 3 | 26 | 34 | 2 |
| df | 0 | 0 | 0 | 9 | 5 | 3 | 0 |
| mel | 0 | 1 | 6 | 0 | 95 | 65 | 0 |
| nv | 1 | 8 | 10 | 2 | 52 | 931 | 1 |
| vasc | 0 | 1 | 0 | 0 | 0 | 2 | 18 |

## Reconciliation

Recomputed from `best_validation_confusion_matrix.csv` before experiment B:

- The CSV matrix equals the matrix in `best_validation_metrics.json`, with
  rows and columns in canonical class order.
- Sample count, accuracy, balanced accuracy, macro precision, recall, F1, and
  every per-class precision, recall, F1, and support match the saved values
  (absolute difference < 1e-12).
- Per-class supports equal the validation-manifest class counts.
- The epoch-9 row of `history.csv` matches the saved metrics, and epoch 9 is
  the arg-max of validation macro F1 over all 14 epochs.

## Observations

- No majority-class collapse: `nv` is predicted for 70.0% of validation
  images against a true share of 67.0%; every class has recall >= 0.53.
- Overfitting from about epoch 5: train loss keeps falling (0.70 to 0.17)
  while validation loss is lowest at epoch 4 (0.536) and rises to about 0.68.
- The main error is melanoma vs. naevus: 65 of 167 melanomas (39%) are
  predicted as `nv`, and 52 naevi as `mel`.

## Limitations

- One seed and one lesion-aware split; no confidence intervals. Validation
  macro F1 moves by 0.02 to 0.04 between neighbouring epochs, so differences
  of that size between experiments may be noise.
- The best epoch was selected on the same validation split that is reported,
  so the best-epoch figures are optimistically biased.
- `df` (17) and `vasc` (21) have very small validation support; one image
  changes their recall by 5 to 6 points.
- Per-class metrics and the confusion matrix are saved for the best epoch
  only; last-epoch values come from `history.csv`.
- Evaluation centre-crops 224 px from a 224 px short side, removing about
  25% of image width; lesion content near the edges may be lost.
- HAM10000 comes from a small number of sources with known demographic and
  acquisition limitations. Results do not indicate clinical usefulness.
