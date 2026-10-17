import os
import time
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset
from torchgeo.datasets import EuroSAT
from torchgeo.models import resnet50, ResNet50_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm


def extract_features(split, model, device, image_size=224, cache_dir="./features"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"eurosat_{split}_{image_size}.pt")

    if os.path.exists(cache_file):
        print(f"Loading cached features: {cache_file}")
        saved = torch.load(cache_file, weights_only=True)
        return saved["X"], saved["y"]

    ds = EuroSAT(root="./data", split=split, download=True)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    feats_list, labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {split} ({image_size}x{image_size})"):
            # TorchGeo SENTINEL2_ALL_MOCO weights expect raw reflectance divided by 10000.0
            imgs = batch["image"].float() / 10000.0

            # Appendix-2 B.1 & Appendix-3 Table XV: resize to 224x224 for better performance
            if image_size != 64:
                imgs = TF.resize(imgs, [image_size, image_size], antialias=True)

            imgs = imgs.to(device)
            feats = model(imgs)
            feats_list.append(feats.cpu())
            labels_list.append(batch["label"])

    X = torch.cat(feats_list, dim=0)
    y = torch.cat(labels_list, dim=0)
    torch.save({"X": X, "y": y}, cache_file)
    return X, y


def train_and_eval_probes(X_train, y_train, X_val, y_val, X_test, y_test):
    # 1. PyTorch Linear Probe (Normalized features + SGD/Cosine or AdamW)
    torch.manual_seed(42)
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-6
    #Zscore normalization
    X_tr = (X_train - mean) / std
    X_va = (X_val - mean) / std
    X_te = (X_test - mean) / std

    classifier = nn.Linear(X_train.shape[1], 10)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(X_tr, y_train), batch_size=256, shuffle=True)

    for _ in range(100):
        classifier.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(classifier(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    classifier.eval()
    with torch.no_grad():
        val_acc_pt = (classifier(X_va).argmax(dim=1) == y_val).float().mean().item() * 100
        test_acc_pt = (classifier(X_te).argmax(dim=1) == y_test).float().mean().item() * 100

    # 2. Scikit-learn L-BFGS Logistic Regression on Train Split (60%)
    pipe_60 = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, C=1.0, random_state=42)
    )
    pipe_60.fit(X_train.numpy(), y_train.numpy())
    test_acc_sk60 = pipe_60.score(X_test.numpy(), y_test.numpy()) * 100

    # 3. Scikit-learn L-BFGS on Train+Val Split (80% = 21,600 images, matching paper split size!)
    X_tr_val = torch.cat([X_train, X_val], dim=0)
    y_tr_val = torch.cat([y_train, y_val], dim=0)
    pipe_80 = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, C=1.0, random_state=42)
    )
    pipe_80.fit(X_tr_val.numpy(), y_tr_val.numpy())
    test_acc_sk80 = pipe_80.score(X_test.numpy(), y_test.numpy()) * 100

    return val_acc_pt, test_acc_pt, test_acc_sk60, test_acc_sk80


def main():
    t0 = time.time()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load frozen ResNet50_Weights.SENTINEL2_ALL_MOCO
    weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    results = {}
    for res in [64, 224]:
        print(f"\n--- Running Evaluation at {res}x{res} Resolution ---")
        X_tr, y_tr = extract_features("train", model, device, image_size=res)
        X_va, y_va = extract_features("val", model, device, image_size=res)
        X_te, y_te = extract_features("test", model, device, image_size=res)

        val_pt, test_pt, test_sk60, test_sk80 = train_and_eval_probes(
            X_tr, y_tr, X_va, y_va, X_te, y_te
        )
        results[res] = (val_pt, test_pt, test_sk60, test_sk80)
        print(f"[{res}x{res}] PyTorch SGD (16.2k train) Val / Test: {val_pt:.2f}% / {test_pt:.2f}%")
        print(f"[{res}x{res}] Sklearn L-BFGS (16.2k train) Test:    {test_sk60:.2f}%")
        print(f"[{res}x{res}] Sklearn L-BFGS (21.6k train+val) Test:{test_sk80:.2f}%")

    mins = (time.time() - t0) / 60
    print("\n" + "=" * 65)
    print("FINAL REPRODUCTION SUMMARY")
    print("=" * 65)
    print("Published Target (Wang et al. Table III & V):       98.00%")
    print(f"Obtained @ 64x64  (Native, 16.2k train, SGD):       {results[64][1]:.2f}%")
    print(f"Obtained @ 224x224 (Upsampled, 16.2k train, SGD):   {results[224][1]:.2f}%")
    print(f"Obtained @ 224x224 (Upsampled, 21.6k train, L-BFGS):{results[224][3]:.2f}%")
    print(f"Total Runtime: {mins:.1f} minutes")
    print("=" * 65)


if __name__ == "__main__":
    main()