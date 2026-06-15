# VLCDC

This repository contains the cleaned core training code for the VLCDC / SG unsupervised person re-identification pipeline.

The main entry point is:

```bash
python cluster-contrast-reid/examples/cluster_contrast_train_usl.py
```

## What Is Kept

- SG model definition and CLIP-based visual/text interaction code.
- Unsupervised clustering training loop.
- Market1501 / MSMT17 style dataset loaders.
- Evaluation, memory bank, re-ranking, and data preprocessing utilities.

Large checkpoints, cached files, logs, old experiment scripts, and unrelated training projects were removed so the repository can be uploaded to GitHub directly.

## Installation

```bash
cd cluster-contrast-reid
pip install -r ../requirements.txt
python setup.py develop
```

## Data

Place datasets under a directory such as:

```text
data/
  market1501/
  MSMT17_V1/
```

Then pass the dataset root with `--data-dir`.

## Training

Example for Market1501:

```bash
python cluster-contrast-reid/examples/cluster_contrast_train_usl.py \
  --dataset market1501 \
  --data-dir ./data \
  --logs-dir ./logs/market1501_sg \
  --gpu 0,1 \
  --eps 0.60 \
  --self-norm \
  --use-hard
```

Example for MSMT17:

```bash
python cluster-contrast-reid/examples/cluster_contrast_train_usl.py \
  --dataset msmt17 \
  --data-dir ./data \
  --logs-dir ./logs/msmt17_sg \
  --gpu 0,1 \
  --eps 0.70 \
  --self-norm \
  --use-hard
```

The default model config is:

```text
cluster-contrast-reid/clustercontrast/configs/person/vit_clipreid.yml
```

## Notes

- Model checkpoints and datasets are intentionally ignored by Git.
- The CLIP backbone is downloaded through the local CLIP loader when needed.
- This code is research code; reproduce the paper environment as closely as possible for stable results.
