# SSL4EO-S12 EuroSAT Reproduction

## Pre-Run Target
- **Published Number**: 98.0% Overall Accuracy (Linear Probing on EuroSAT, all 13 bands).
- **Source**: Wang et al. (2023), *SSL4EO-S12* (arXiv:2211.07044v2), Table III & Table V (page 4), ResNet-50 (RN50) pretrained with MoCo-v2.
- **Reproduction Criteria**: Within ±1.5% of the published number (i.e., ≥ 96.5% test accuracy), allowing for minor differences in random seeds, train/test splits, and linear classifier hyperparameters.