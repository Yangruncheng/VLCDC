<img width="1371" height="489" alt="image" src="https://github.com/user-attachments/assets/4f022fd9-3bf9-4c85-8790-b088895fd553" />



# VLCDC: Vision-Language Complementary Dual-Centroid Learning for Unsupervised Person Re-Identification (PR2026) 

This repository provides the cleaned core implementation of **VLCDC**, a vision-language complementary dual-centroid learning framework for **unsupervised person re-identification**.

> **Vision-Language Complementary Dual-Centroid Learning for Unsupervised Person Re-Identification**
> Runcheng Yang, Guorong Lin, Chang-Dong Wang, Xiaowen Ma, Zhenhua Huang
> *Pattern Recognition*, 2026
> DOI: [10.1016/j.patcog.2026.114278](https://doi.org/10.1016/j.patcog.2026.114278)

## Overview

Unsupervised person re-identification aims to learn discriminative pedestrian representations without identity annotations. Most existing methods rely on clustering-based pseudo labels and visual centroid memories. However, visual centroids can be sensitive to pose changes, viewpoint variations, occlusions, and illumination differences, which may introduce noisy pseudo labels and unstable supervision.

VLCDC addresses this issue by introducing **textual centroids** to complement conventional visual centroids. By leveraging the vision-language alignment ability of CLIP, VLCDC constructs visual-textual complementary prototypes for more robust unsupervised representation learning.

The framework mainly contains three components:

* **Key Part-aware Attention (KPA)**: adaptively selects discriminative local patches from pedestrian images.
* **Complementary Textual Prompting (CTP)**: combines global and local visual cues to construct informative textual prompts and textual centroids.
* **Appearance Ambiguity Enhancement (AAE)**: applies saliency-guided structured occlusion to encourage the model to learn more robust identity-discriminative features.
<img width="1061" height="702" alt="image" src="https://github.com/user-attachments/assets/65356b58-02ed-4b3c-9771-a15106f933d9" />
The main training entry point is:

```bash
python cluster-contrast-reid/examples/cluster_contrast_train_usl.py
```

## Repository Structure

```text
VLCDC/
  cluster-contrast-reid/
    clustercontrast/
      configs/
      datasets/
      evaluation/
      models/
      trainers/
      utils/
    examples/
      cluster_contrast_train_usl.py
  requirements.txt
  README.md
```

This repository keeps the core research code required for training and evaluation, including:

* CLIP-based visual encoder and vision-language interaction modules.
* KPA, CTP, and AAE related implementation.
* Unsupervised clustering-based training pipeline.
* Market-1501, DukeMTMC-reID, and MSMT17 style dataset loaders.
* Memory bank, re-ranking, evaluation, and preprocessing utilities.

Large checkpoints, cached files, logs, old experiment scripts, and unrelated training projects are intentionally removed to keep the repository clean.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Yangruncheng/VLCDC.git
cd VLCDC

pip install -r requirements.txt
cd cluster-contrast-reid
python setup.py develop
```

## Datasets

Please prepare the datasets manually and place them under a directory such as:

```text
data/
  market1501/
  DukeMTMC-reID/
  MSMT17_V1/
```

Then specify the dataset root using `--data-dir`.

Example:

```bash
--data-dir ./data
```

The expected dataset names are:

* `market1501`
* `dukemtmc`
* `msmt17`

Please make sure the dataset folder structure is consistent with the corresponding dataset loader in `clustercontrast/datasets/`.

## Training

### Market-1501

```bash
python cluster-contrast-reid/examples/cluster_contrast_train_usl.py \
  --dataset market1501 \
  --data-dir ./data \
  --logs-dir ./logs/market1501_vlcdc \
  --gpu 0,1 \
  --eps 0.60 \
  --self-norm \
  --use-hard
```

### DukeMTMC-reID

```bash
python cluster-contrast-reid/examples/cluster_contrast_train_usl.py \
  --dataset dukemtmc \
  --data-dir ./data \
  --logs-dir ./logs/dukemtmc_vlcdc \
  --gpu 0,1 \
  --eps 0.60 \
  --self-norm \
  --use-hard
```

### MSMT17

```bash
python cluster-contrast-reid/examples/cluster_contrast_train_usl.py \
  --dataset msmt17 \
  --data-dir ./data \
  --logs-dir ./logs/msmt17_vlcdc \
  --gpu 0,1 \
  --eps 0.70 \
  --self-norm \
  --use-hard
```

### Experiment
<img width="534" height="200" alt="image" src="https://github.com/user-attachments/assets/fa8d634c-b6ea-41a5-a0a9-0ba60ad0e631" />


<img width="934" height="349" alt="image" src="https://github.com/user-attachments/assets/c1efb4f3-34a6-4e0a-86d9-f9671e3bcce6" />











## Configuration

The default model configuration is located at:

```text
cluster-contrast-reid/clustercontrast/configs/person/vit_clipreid.yml
```

You may modify this file to adjust the CLIP backbone, image size, model settings, or other training-related parameters.

## Notes

* Model checkpoints and datasets are not included in this repository.
* The CLIP backbone will be loaded through the local CLIP loader when needed.
* For stable reproduction, please keep the training environment, dataset structure, and hyperparameters as close as possible to those used in the paper.
* This repository is research code and may require minor path or environment adjustments on different machines.

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{yang2026vlcdc,
  title   = {Vision-Language Complementary Dual-Centroid Learning for Unsupervised Person Re-Identification},
  author  = {Yang, Runcheng and Lin, Guorong and Wang, Chang-Dong and Ma, Xiaowen and Huang, Zhenhua},
  journal = {Pattern Recognition},
  year    = {2026},
  doi     = {10.1016/j.patcog.2026.114278}
}
```

## Acknowledgements

This codebase is built upon the Cluster Contrast framework and CLIP-related implementations. We sincerely thank the authors of the open-source projects that made this research possible.

## Contact

For questions or discussions, please contact:

```text
Runcheng Yang
Email: 2024025447@m.scnu.edu.cn
GitHub: https://github.com/Yangruncheng
```
