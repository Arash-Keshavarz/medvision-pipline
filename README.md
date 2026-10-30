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

The accepted baseline (experiment A) is summarised in
`reports/experiments/experiment_a_baseline_resnet18.md`.

### Class-weighted cross-entropy (experiment B)

`configs/resnet18_weighted_ce.yaml` is identical to the baseline config
except for `training.loss: weighted_cross_entropy` and its output directory.
Weights come from the training manifest only, in canonical class order:

    weight[c] = N_train / (number_of_classes * count[c])

A class with no training images is rejected. Counts, formula, and weights
are saved to `class_weights.json` in the run directory (and in
`resolved_config.json` and the checkpoints).

With weighted cross-entropy, PyTorch's mean reduction divides by the sum of
the targets' weights, so epoch losses are aggregated the same way and equal
the loss over the whole epoch in one batch. These weighted losses are not
comparable with the baseline's; `history.csv` also records
`*_unweighted_cross_entropy` columns for that. Model selection still uses
validation macro F1.

```bash
python scripts/train_baseline.py --config configs/resnet18_weighted_ce.yaml
```

The comparison with experiment A is in
`reports/experiments/experiment_b_weighted_ce_vs_a.md`.

### Experiment tracking (MLflow, optional)

Tracking is off by default and needs the `mlops` extra. Install it with the
tested MLflow version (3.17.2) recorded in `constraints-mlops.txt`:
`pip install -e '.[mlops]' -c constraints-mlops.txt`. Other versions allowed
by `pyproject.toml` are untested. It is local only: an SQLite database and
artifact directory under `mlruns/` (ignored by Git), with no credentials or
remote server, and with MLflow telemetry disabled.

```bash
# Track a new training run (local run directory is written as usual).
python scripts/train_baseline.py --config configs/baseline_resnet18.yaml --mlflow

# Import finished run directories (reads saved files only; no training).
python scripts/import_mlflow_runs.py runs/baseline_resnet18/<run-id> ...

# Browse runs at http://127.0.0.1:5000
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000
```

Each MLflow run records the following:

- configuration, seeds, Git commit and dirty state, manifest hashes,
  environment, and class-weight provenance;
- per-epoch metrics (step = epoch);
- `best.*` (selected epoch) and `final.*` (last epoch) summaries;
- small run files and diagnostic plots.

Smoke runs are tagged `medvision.smoke=true`. Imported runs keep the original
run name, start and end times, and source commit
(`mlflow.source.git.commit`); the importer's commit is recorded separately,
and importing the same run twice is refused.

Checkpoints are **not** copied into MLflow. `checkpoints/checkpoint_reference.json`
records each checkpoint's local path, size, and SHA-256. That path is valid
only on the machine where the run was saved; it is not a portable model
artifact. The run directory remains the source of truth.

If tracking fails during training, the following happens:

- training continues;
- a warning is printed, and an attempt is made to mark the MLflow run `FAILED`;
- the failure is recorded in `run_metadata.json` (`tracking.mlflow_status`
  and `tracking.error`);
- the script exits non-zero.

If tracking cannot start, training does not begin.

`run_metadata.json` is finalised (run status `completed`, `failed`, or
`interrupted`, and the expected MLflow status) before it is logged, so the
MLflow copy matches the local file when logging succeeds.

If an MLflow backend operation fails (for example while logging the final
files or closing the run), the local `run_metadata.json` records the error
and `tracking.mlflow_status: FAILED`. Marking the MLflow run `FAILED` is only
attempted. If the backend is still failing, the MLflow run may stay
`RUNNING` or keep an older status and copy of the metadata. The local file is
authoritative, and the MLflow status may need manual reconciliation.

**Recovery keeps failed runs as history.** Do not delete them by default.
The importer refuses a run directory that is already in the target
experiment, including when that run is `FAILED`. This is a current
limitation. To record the complete run without deleting anything, import it
into a separate experiment:

```bash
python scripts/import_mlflow_runs.py runs/<output-root>/<run-id> \
    --mlflow-experiment ham10000-classification-recovered
```

The `FAILED` run stays in the original experiment, tagged
`medvision.tracking_error`.
