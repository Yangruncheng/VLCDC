# VLC SG ReID

This repository contains the cleaned SG training code used for unsupervised person re-identification experiments.

The active training entry is:

```bash
cluster-contrast-reid/examples/cluster_contrast_train_usl.py
```

## Environment

```bash
pip install -r requirements.txt
cd cluster-contrast-reid
python setup.py develop
```

The original experiments used PyTorch 1.7.x, torchvision 0.8.x, timm 0.3.x, CUDA, and faiss-gpu.

## Data

The dataset loaders currently support Market-1501 and MSMT17. Update the dataset paths in:

```bash
cluster-contrast-reid/clustercontrast/datasets/market1501.py
cluster-contrast-reid/clustercontrast/datasets/msmt17.py
```

or pass a root prefix with `--data-dir` if you keep the same internal folder layout.

## Training

Market-1501:

```bash
cd cluster-contrast-reid
bash market_usl.sh
```

MSMT17:

```bash
cd cluster-contrast-reid
bash msmt_usl.sh
```

The SG model configuration is stored in:

```bash
cluster-contrast-reid/clustercontrast/configs/person/vit_clipreid.yml
```

Checkpoints and logs are written to the path specified by `--logs-dir`.

## Model Weights

Model weights and generated checkpoints are not included in this repository. Please download pretrained weights separately and keep them outside the source tree or in a local ignored directory.

## Repository Scope

This cleaned version keeps the SG training path, Cluster Contrast memory, dataset loaders, evaluation metrics, and CLIP model components. Legacy pretraining, supervised TransReID, exploratory visualization, cached bytecode, temporary analysis scripts, and large model weights were removed.
