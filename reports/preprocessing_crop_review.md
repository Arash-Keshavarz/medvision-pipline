# Preprocessing crop review (exploratory)

Date: 2026-10-30

This is an **exploratory** check of how much lesion content the cropping
steps remove. It is not a validated or representative estimate:

- Lesion areas come from a heuristic mask (darkness threshold with Otsu's
  method, dark vignette pixels excluded, connected component nearest the
  image centre). Visual inspection showed the mask failing on some faint or
  diffuse lesions, where it traced skin texture or the vignette instead.
- The sample was 600 images, 150 per source. This balances sources rather
  than following the dataset's source and class distribution.
- The masking code was a one-off review script. It is deliberately not part
  of the Dataset or the production transforms.

## Evaluation transform

On 600x450 images, evaluation resizes the shorter side to `resize_size` and
centre-crops 224x224. Approximate share of the lesion mask that was kept:

| Setting | Image area kept | Median | 10th percentile | Images losing >10% of lesion |
|---|---|---|---|---|
| `resize_size: 256` (previous) | 57% (66% width, 88% height) | 100% | 73% | 21.5% |
| `resize_size: 224` (adopted) | 75% (75% width, full height) | 100% | 84% | 13.5% |

Decision: `resize_size` was changed to 224. Evaluation still centre-crops the
sides of the image, removing about 25% of the original width in total (12.5%
from each side), so lesions that extend sideways can lose part of their
border. About a fifth of the masks already touched the image
edge in the original photo, where no crop can recover the full lesion.

## Training transform

Unchanged for the baseline: rotation up to ±10° with a border-free inscribed
crop, then `RandomResizedCrop(scale=(0.7, 1.0), ratio=(3/4, 4/3))` and
flips. The median lesion share kept per image was about 100%; about 12% of
random draws kept less than 80% of the lesion, rising to about 25% for large
lesions (mask covering more than 30% of the image). This partial-view
augmentation was judged acceptable.

## Follow-up

Compare an aspect-preserving resize-and-pad evaluation transform against the
current centre crop in a controlled experiment after the baseline exists.
