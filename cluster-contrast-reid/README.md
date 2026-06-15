# SG Cluster Contrast ReID

This package contains the SG unsupervised ReID training pipeline.

## Install

```bash
python setup.py develop
```

## Train

```bash
bash market_usl.sh
bash msmt_usl.sh
```

The main entry is:

```bash
python examples/cluster_contrast_train_usl.py
```

Useful options:

```bash
--dataset market1501
--dataset msmt17
--config-file clustercontrast/configs/person/vit_clipreid.yml
--logs-dir logs/market1501_sg
--eps 0.6
--num-instances 8
--self-norm
--use-hard
```

## Notes

The current public code is intentionally focused on the SG model path. Removed files include old cache folders, exploratory visualizations, unused test scripts, DINO pretraining code, TransReID supervised training code, and large model weights. Download pretrained weights separately instead of committing them to this repository.
