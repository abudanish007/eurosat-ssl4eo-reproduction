# SSL4EO-S12 EuroSAT Reproduction

## Pre-Run Target

- **Published Number**: 98.0% Overall Accuracy (Linear Probing on EuroSAT, all 13 bands).
- **Source**: Wang et al. (2023), *SSL4EO-S12* (arXiv:2211.07044v2), Table III & Table V (page 4), ResNet-50 (RN50) pretrained with MoCo-v2.
- **Reproduction Criteria**: Within ±1.5% of the published number (i.e., ≥ 96.5% test accuracy), allowing for minor differences in random seeds, train/test splits, and linear classifier hyperparameters.

---

## How to Run

Install dependencies and run with a single command:

```bash
pip install -r requirements.txt
python reproduce.py
```

Extracted features are cached in `./features/` after the first pass. Total runtime is ~7.7 minutes on Apple Silicon (`mps`).

---

## Results & Gap Analysis

| Evaluation Setting | Train Samples | Solver | Val Acc | Test Acc | Gap to Paper (98.0%) |
|---|---|---|---|---|---|
| **Paper Target** (Wang et al., Table III & V) | 21,600 | SGD (100 ep, online aug) | — | 98.00% | — |
| Native 64x64 (TorchGeo default split) | 16,200 | PyTorch SGD (100 ep) | 87.19% | 86.46% | -11.54% |
| Native 64x64 (TorchGeo default split) | 16,200 | Sklearn L-BFGS | — | 85.94% | -12.06% |
| Upsampled 224x224 (TorchGeo default split) | 16,200 | PyTorch SGD (100 ep) | 95.69% | 95.83% | -2.17% |
| Upsampled 224x224 (TorchGeo default split) | 16,200 | Sklearn L-BFGS | — | 95.67% | -2.33% |
| **Upsampled 224x224 (Paper-matched 80/20 size)** | 21,600 | Sklearn L-BFGS | — | **96.00%** | **-2.00%** |

---

## Why I Believe This Result & What Explains the Gap

Our best static-feature linear probe reached **96.00%** test accuracy, falling just 0.50% short of our pre-run reproduction threshold (96.5%) and 2.00% below the paper's 98.0%. Investigating the evaluation setup in the paper explains this progression:

1. **Input Resolution (64x64 vs. 224x224), accounts for +9.37%**
   Running the frozen backbone on native 64x64 EuroSAT patches initially yielded only 86.46%. SSL4EO-S12 pre-trained ResNet-50 on 224x224 crops, so a 64x64 input shrinks to a 2x2 feature map prior to global average pooling (instead of 7x7). Upsampling EuroSAT images to 224x224 to match the paper's evaluation setup increased test accuracy from 86.46% to 95.83%.

2. **Train/Test Split Size (16,200 vs. 21,600 train images), accounts for +0.33%**
   The paper used an 80/20 split (21,600 train / 5,400 test), whereas TorchGeo's standard EuroSAT dataset uses a 60/20/20 split (16,200 train / 5,400 val / 5,400 test), which is 25% less training data. Combining TorchGeo's train + val splits (21,600 images) and evaluating on the untouched test split pushed L-BFGS accuracy from 95.67% to 96.00%.

3. **Static Frozen Features vs. 100-Epoch Online Augmentation, explains the remaining 2.00% gap**
   Per the task instructions, we froze the backbone, extracted one static feature vector per unaugmented image, and trained a linear classifier on top. In contrast, the paper trained the linear layer live for 100 epochs with random resized crops and horizontal flips applied at every epoch. Training on 100 random augmented views per image acts as strong regularization and explains why static feature extraction plateaus at 96.00% rather than 98.00%.

### Verification Checks

- Verified reflectance scaling (`image / 10000.0`) matches `SENTINEL2_ALL_MOCO` pre-training expectations.
- Cross-checked PyTorch SGD (100 epochs, step decay at epochs 60 and 80) against exact convex optimization (scikit-learn L-BFGS with `StandardScaler`). Both converged to within 0.16% of each other (95.83% vs. 95.67%), confirming the linear classifier itself is well-tuned.

---

## Time Spent & Tools Used

- **Time Taken**: ~4–5 hours total (reading the paper, environment setup, feature extraction runs, and gap analysis).
- **Tools Used**: Python 3, PyTorch (Apple Silicon `mps`), TorchGeo, Scikit-Learn, Git, and Gemini (AI assistant) for structuring the feature-caching script and organizing the analysis.