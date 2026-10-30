# MedVision Pipeline

An end-to-end medical image classification system for multi-class skin
lesion analysis.

The project is designed to demonstrate the complete machine-learning
lifecycle: data validation, reproducible model training, evaluation,
inference serving, prediction logging, containerization, testing, and
deployment.

> This project is intended for research and educational purposes only.
> Its models are research and portfolio baselines, not clinical
> diagnostic systems. It is not a medical device and must not be used
> for clinical diagnosis.

## Project goals

The first release will:

- Train a PyTorch baseline for seven-class skin lesion classification.
- Build reproducible, lesion-aware train, validation, and test splits.
- Detect duplicate images before splitting the dataset.
- Evaluate performance using balanced accuracy, macro F1, per-class
  recall, ROC-AUC, and a confusion matrix.
- Serve the trained model through a FastAPI inference API.
- Store prediction results and model versions in PostgreSQL.
- Run the API and database using Docker Compose.
- Include automated tests and continuous integration.

Later releases will introduce experiment tracking, data versioning,
Vision Transformers, explainability, metadata fusion, and cloud
deployment.

## Dataset

The project uses the official
[HAM10000 dataset](https://doi.org/10.7910/DVN/DBW86T)
(Tschandl et al., Harvard Dataverse): 10,015 dermoscopic images of
7,470 unique lesions across seven diagnosis classes (`akiec`, `bcc`,
`bkl`, `df`, `mel`, `nv`, `vasc`).

The images are not stored in this repository. They are expected locally
under `data/raw/ham10000/` (`HAM10000_metadata.csv` and `images/`).

### Dataset history

The first prototype used a third-party Kaggle ISIC dataset. It was
rejected after the duplicate audit found byte-identical images with
conflicting class labels and images shared between its train and test
folders.

HAM10000 passed file and metadata validation: every image is readable,
every image has exactly one metadata record, and every metadata record
has an image. See `reports/ham10000_audit/`,
`reports/ham10000_duplicates/`, and `notebooks/01_EDA_HAM.ipynb` for
the audit, duplicate check, and exploratory analysis.

### Splitting

Many lesions have several images, so splits are created by `lesion_id`
rather than by image: no lesion appears in more than one split. HAM10000
does not provide patient identifiers, so the evaluation is
lesion-independent but not guaranteed to be patient-independent.

## Planned releases

| Version | Scope |
|---|---|
| `v0.1.0` | CNN baseline, evaluation, FastAPI, PostgreSQL and Docker |
| `v0.2.0` | MLflow, DVC and reproducible experiment management |
| `v0.3.0` | Controlled CNN versus Vision Transformer comparison |
| `v0.4.0` | Explainability and metadata fusion |
| `v0.5.0` | AWS deployment and monitoring |
| `v1.0.0` | Polished portfolio release |

## Repository structure

```text
medvision-pipeline/
├── api/
├── configs/
├── data/
├── database/
├── notebooks/
├── reports/
├── scripts/
├── src/
│   └── medvision/
│       ├── data/
│       ├── evaluation/
│       ├── inference/
│       ├── models/
│       └── training/
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

### Evaluation approach

HAM10000 is strongly imbalanced (`nv` is about two thirds of all
images). The project therefore:

- uses balanced accuracy and macro F1 as primary metrics;
- reports per-class precision, recall, F1, and support;
- reports class and source distributions for every split.

### Baseline training

The ResNet18 baseline fine-tunes all layers from ImageNet weights with
ordinary cross-entropy (no class weights or weighted sampling) and AdamW.
The checkpoint with the highest validation macro F1 is kept, and training
stops after five epochs without improvement. Only the train and validation
splits are used; the test split is not loaded during training.

Settings come from two files: `configs/baseline_resnet18.yaml` (model,
optimiser, run) and `configs/data.yaml` (paths, preprocessing, loaders).

```bash
# Offline engineering check: random weights, one epoch, small subsets.
python scripts/train_baseline.py --smoke

# Full baseline training.
python scripts/train_baseline.py --config configs/baseline_resnet18.yaml
```

Each run writes to a new directory under `runs/baseline_resnet18/`
(ignored by Git). A fixed seed makes runs repeatable on the same device and
library versions, but not bitwise identical across devices.
