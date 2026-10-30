# Dataset

Dataset files are stored locally and are not tracked by Git.

## Initial dataset

The MVP uses the
[Skin Cancer ISIC dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
published on Kaggle.

The dataset contains 2,357 dermoscopic images organized into nine
skin-lesion classes:

1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion

The dataset is used only for research, education, and portfolio
development. This project is not intended for clinical diagnosis.

## Authentication

Install the optional data dependencies:

```bash
pip install -e ".[data]"

kaggle datasets download \
  nodoubttome/skin-cancer9-classesisic \
  --path data/raw/skin-cancer-isic \
  --unzip
```