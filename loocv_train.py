"""
Leave-One-Out Cross-Validation for MaskGNN
------------------------------------------
For each fold:
  • Leave one subject out as test
  • Train on remaining subjects
  • Save:
        best_mask_fold_{i}.npy
        best_maskgnn_fold_{i}.pt
  • Save predictions in loocv_results.csv

Run:
  python loocv_train.py --graphs graphs_prepared --epochs 50 --scale-features --scale-labels
"""

import os, glob, math, random, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv

# -------------------------
#  MaskGNN Model
# -------------------------
class MaskGNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(in_feats, out_feats) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_feats))
        self.act = nn.ReLU()

    def forward(self, H, G):
        out = G @ H @ self.theta + self.bias
        return self.act(out)

class MaskGNN(nn.Module):
    def __init__(self, Q, in_dim, hidden=[64, 64], dropout=0.5):
        super().__init__()
        self.Q = Q
        self.V = nn.Parameter(torch.randn(Q, Q) * 0.01)
        dims = [in_dim] + hidden
        self.layers = nn.ModuleList([MaskGNNLayer(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(nn.Linear(dims[-1], 128), nn.ReLU(), nn.Linear(128, 1))

    def build_mask(self):
        return torch.sigmoid(self.V + self.V.T)

    def normalize_adj(self, A):
        I = torch.eye(A.shape[0], device=A.device)
        A = A + I
        deg = A.sum(dim=1)
        deg_inv = torch.pow(deg + 1e-12, -0.5)
        D = torch.diag(deg_inv)
        return D @ A @ D

    def forward(self, x, A):
        M = self.build_mask()
        A_norm = self.normalize_adj(A)
        G = (M + torch.eye(self.Q, device=A.device)) * A_norm

        H = x
        for layer in self.layers:
            H = layer(H, G)
            H = self.dropout(H)

        g_emb = H.mean(dim=0)
        out = self.readout(g_emb)
        return out.squeeze(), H, M


# -------------------------
#  Utilities
# -------------------------
def load_graphs(folder):
    items = []
    for path in sorted(glob.glob(os.path.join(folder, "*.pt"))):
        g = torch.load(path, map_location="cpu")
        sid = Path(path).stem
        x, adj, y = g["x"], g["adj"], g["y"]

        if torch.is_tensor(x): x = x.numpy()
        if torch.is_tensor(adj): adj = adj.numpy()
        if torch.is_tensor(y): y = float(y.numpy().reshape(-1)[0])

        items.append({"sid": sid, "x": x, "adj": adj, "y": y})
    return items

def minmax_scale_adj(A):
    Q = A.shape[0]
    tri = np.triu_indices(Q, 1)

    vals = A[tri].reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vals).flatten()

    M = np.zeros_like(A)
    M[tri] = scaled
    return M + M.T

def keep_topk(A, K):
    Q = A.shape[0]
    A2 = np.zeros_like(A)
    for i in range(Q):
        row = A[i].copy()
        row[i] = 0
        idx = np.argsort(row)[::-1][:K]
        A2[i, idx] = A[i, idx]
    return np.maximum(A2, A2.T)

def manifold_loss(H, A):
    D = torch.diag(A.sum(dim=1))
    L = D - A
    HL = H.T @ L @ H
    return torch.trace(HL)

def mask_penalty(M, l1, l2, o):
    l1_term = torch.norm(M, p=1)
    l2_term = torch.norm(M, p='fro') ** 2
    orth = torch.norm(M @ M.T - torch.eye(M.shape[0], device=M.device), p='fro') ** 2
    return l1*l1_term + l2*l2_term + o*orth


# -------------------------
#  LOOCV Master Loop
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", default="graphs_prepared")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--topk", default=30, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--mask-l1", default=1e-4, type=float)
    parser.add_argument("--mask-l2", default=1e-4, type=float)
    parser.add_argument("--mask-orth", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--scale-features", action="store_true")
    parser.add_argument("--scale-labels", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cfg = parser.parse_args()

    device = torch.device(cfg.device)
    print("Using device:", device)

    # Load data
    items = load_graphs(cfg.graphs)
    labeled = [it for it in items if it["y"] is not None]

    print(f"Loaded {len(items)} graphs; {len(labeled)} with labels.")

    results = []

    # LOOCV folds
    for i in range(len(labeled)):
        test = labeled[i]
        train = [labeled[j] for j in range(len(labeled)) if j != i]

        print(f"\n=== Fold {i+1}/{len(labeled)}: test={test['sid']} train={len(train)} ===")

        # Optional feature scale
        if cfg.scale_features:
            scaler = MinMaxScaler()
            stacked = np.vstack([t["x"] for t in train])
            scaler.fit(stacked)
            for t in train + [test]:
                t["x"] = scaler.transform(t["x"])

        # Optional label scale
        label_mean, label_std = 0, 1
        if cfg.scale_labels:
            ys = np.array([t["y"] for t in train])
            label_mean, label_std = ys.mean(), ys.std() if ys.std()>0 else 1
            for t in train + [test]:
                t["y"] = (t["y"] - label_mean)/label_std

        # Build model
        Q = train[0]["x"].shape[0]
        in_dim = train[0]["x"].shape[1]
        model = MaskGNN(Q, in_dim).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        # Training
        best_val = float("inf")
        best_state = None

        for epoch in range(cfg.epochs):
            model.train()
            losses = []
            for item in train:
                x = torch.tensor(item["x"], dtype=torch.float32, device=device)
                A = keep_topk(minmax_scale_adj(item["adj"]), cfg.topk)
                A = torch.tensor(A, dtype=torch.float32, device=device)
                y = torch.tensor(float(item["y"]), device=device)

                optim.zero_grad()
                pred, H, M = model(x, A)
                mse = F.mse_loss(pred, y)

                A_masked = (M + torch.eye(Q, device=device)) * model.normalize_adj(A)
                reg_man = manifold_loss(H, A_masked)
                reg_mask = mask_penalty(M, cfg.mask_l1, cfg.mask_l2, cfg.mask_orth)

                loss = mse + cfg.alpha*reg_man + reg_mask
                loss.backward()
                optim.step()
                losses.append(loss.item())

            val_rmse = math.sqrt(F.mse_loss(pred, y).item())

            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

        # Evaluate on test subject
        model.load_state_dict(best_state)
        model.eval()

        xt = torch.tensor(test["x"], dtype=torch.float32, device=device)
        At = keep_topk(minmax_scale_adj(test["adj"]), cfg.topk)
        At = torch.tensor(At, dtype=torch.float32, device=device)

        with torch.no_grad():
            pred, H, M = model(xt, At)
            pred = float(pred.cpu().numpy())
            true = float(test["y"])

            if cfg.scale_labels:
                pred = pred * label_std + label_mean
                true = true * label_std + label_mean

        abs_err = abs(pred - true)
        print(f"Test pred={pred:.3f}, true={true:.3f}, abs_err={abs_err:.3f}")

        # Save per-fold mask
        bestM = M.cpu().numpy()
        np.save(f"best_mask_fold_{i+1}.npy", bestM)
        torch.save(best_state, f"best_maskgnn_fold_{i+1}.pt")

        results.append({"fold": i+1, "sid": test["sid"], "true": true, "pred": pred, "abs": abs_err})

    # Save CSV
    with open("loocv_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["fold", "sid", "true", "pred", "abs"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    maes = [r["abs"] for r in results]
    print("\n=== LOOCV SUMMARY ===")
    print("Mean MAE =", np.mean(maes))
    print("Std MAE =", np.std(maes))
    print("Saved loocv_results.csv and best_mask_fold_*.npy")
    
if __name__ == "__main__":
    main()
