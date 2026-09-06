# MedVision Pipeline

An end-to-end medical image classification system for multi-class skin
lesion analysis.

The project is designed to demonstrate the complete machine-learning
lifecycle: data validation, reproducible model training, evaluation,
inference serving, prediction logging, containerization, testing, and
deployment.

> This project is intended for research and educational purposes only.
> It is not a medical device and must not be used for clinical diagnosis.

## Project goals

The first release will:

- Train a PyTorch baseline for nine-class skin lesion classification.
- Build reproducible train, validation, and test splits.
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

The initial MVP uses the
[Skin Cancer ISIC dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic).

It contains 2,357 dermoscopic images organized into nine skin-lesion
classes:

1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion

The images are not stored in this repository. See `data/README.md` for
download and preparation instructions.

Because this curated dataset does not provide reliable patient
identifiers, the MVP uses duplicate-aware stratified splitting and does
not claim patient-independent evaluation.

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

### Dataset audit findings

The dataset contains 2,357 readable RGB JPEG images across nine
skin-lesion classes:

- 2,239 training images
- 118 test images
- 0 corrupted or unreadable images

Both splits have the same median image resolution of 600 x 450 pixels
and a median aspect ratio of 4:3. However, the test split contains
several higher-resolution outliers, reaching a maximum resolution of
6688 x 4479 pixels.

The training split is imbalanced. The largest class contains 462 images,
while the smallest contains 77 images, resulting in an imbalance ratio
of 6:1.

The provided test set is small. Most classes contain 16 test images,
while seborrheic keratosis and vascular lesion contain only three test
images each. Consequently, per-class test metrics for these classes
will have high uncertainty.

The project will therefore:

- use balanced accuracy and macro F1 as primary metrics;
- report per-class recall and support;
- investigate weighted loss and weighted sampling;
- preserve the provided test set as a final holdout;
- create a validation split only from the original training data;
- perform duplicate detection before creating new split manifests.

