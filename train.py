# train.py (patched)
"""
Train MaskGNN on prepared .pt graphs (graphs_prepared/)
Saves:
  - best_maskgnn.pt  (state_dict)
  - best_mask.npy    (learned mask matrix)
Usage example:
  python train.py --graphs graphs_prepared --epochs 50 --lr 0.005 --topk 30 --seed 42 --scale-features --scale-labels
"""

import os
import glob
import argparse
import random
import math
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# Utilities: seeding & device
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic for reproducibility (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(prefer_cuda: bool):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# -----------------------------
# Data loader / preprocessing
# -----------------------------
def load_graphs_from_dir(folder: str) -> List[Dict[str, Any]]:
    items = []
    for p in sorted(glob.glob(os.path.join(folder, "*.pt"))):
        g = torch.load(p)
        # support dict or object
        if isinstance(g, dict):
            x = g.get("x", None)
            adj = g.get("adj", None)
            y = g.get("y", None)
        else:
            x = getattr(g, "x", None)
            adj = getattr(g, "adj", None)
            y = getattr(g, "y", None)
        # convert tensors to numpy
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        if torch.is_tensor(adj):
            adj = adj.cpu().numpy()
        if torch.is_tensor(y):
            try:
                y = float(y.cpu().numpy().reshape(-1)[0])
            except:
                y = None
        sid = Path(p).stem
        items.append({"sid": sid, "x": x, "adj": adj, "y": y, "path": p})
    return items

def minmax_scale_adj(A: np.ndarray) -> np.ndarray:
    """Min-max scale upper-triangular edges to [0,1] and symmetrize."""
    Q = A.shape[0]
    tri = np.triu_indices(Q, k=1)
    vals = A[tri].reshape(-1, 1)
    if np.allclose(vals, vals[0]):
        scaled = np.zeros_like(vals)
    else:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(vals).flatten()
    M = np.zeros_like(A)
    M[tri] = scaled
    M = M + M.T
    return M

def keep_topk_per_row(A: np.ndarray, K: int) -> np.ndarray:
    Q = A.shape[0]
    A2 = np.zeros_like(A)
    for i in range(Q):
        row = A[i].copy()
        row[i] = 0
        idx = np.argsort(row)[::-1]  # descending
        top_idx = idx[:K]
        A2[i, top_idx] = A[i, top_idx]
    # symmetrize taking max
    return np.maximum(A2, A2.T)

# -----------------------------
# Model: MaskGNN
# -----------------------------
class MaskGNNLayer(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, bias: bool = True):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(in_feats, out_feats) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_feats)) if bias else None
        self.act = nn.ReLU()

    def forward(self, H: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        # H: (Q, d), G: (Q, Q)
        out = G @ H @ self.theta
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)

class MaskGNN(nn.Module):
    def __init__(self, Q: int, in_dim: int, hidden_dims: List[int] = [64, 64], dropout: float = 0.5):
        super().__init__()
        self.Q = Q
        # learnable parameter V (will produce symmetric M)
        self.V = nn.Parameter(torch.randn(Q, Q) * 0.01)
        dims = [in_dim] + hidden_dims
        self.layers = nn.ModuleList([MaskGNNLayer(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(nn.Linear(dims[-1], 128), nn.ReLU(), nn.Linear(128, 1))

    def build_mask(self) -> torch.Tensor:
        return torch.sigmoid(self.V + self.V.T)

    def normalize_adj(self, A: torch.Tensor) -> torch.Tensor:
        # expects A: (Q,Q)
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        A_tilde = A + I
        deg = A_tilde.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def forward(self, x: torch.Tensor, A_base: torch.Tensor):
        # x: (Q, in_dim), A_base: (Q,Q)
        M = self.build_mask()  # (Q,Q)
        A_norm = self.normalize_adj(A_base)  # (Q,Q)
        G = (M + torch.eye(self.Q, device=A_norm.device)) * A_norm
        H = x
        for layer in self.layers:
            H = layer(H, G)
            H = self.dropout(H)
        graph_emb = H.mean(dim=0)
        out = self.readout(graph_emb)
        return out.squeeze(), H, M

# -----------------------------
# Loss helpers
# -----------------------------
def manifold_loss(H: torch.Tensor, A_masked: torch.Tensor) -> torch.Tensor:
    # H: Qxd, A_masked: QxQ
    D = torch.diag(A_masked.sum(dim=1))
    L = D - A_masked
    HL = H.T @ L @ H
    return torch.trace(HL)

def mask_penalty(M: torch.Tensor, l1: float, l2: float, lambda_orth: float) -> torch.Tensor:
    l1_term = torch.norm(M, p=1)
    l2_term = torch.norm(M, p='fro') ** 2
    orth_term = torch.norm(M @ M.T - torch.eye(M.shape[0], device=M.device), p='fro') ** 2
    return l1 * l1_term + l2 * l2_term + lambda_orth * orth_term

# -----------------------------
# Training / evaluation loops
# -----------------------------
def eval_model(model: MaskGNN, items: List[dict], device: torch.device, cfg: argparse.Namespace, label_mean=0.0, label_std=1.0) -> dict:
    model.eval()
    ys = []
    preds = []
    with torch.no_grad():
        for item in items:
            if item['y'] is None:
                continue
            x = torch.from_numpy(item['x']).to(device)
            A_proc = minmax_scale_adj(item['adj'])
            A_proc = keep_topk_per_row(A_proc, K=cfg.topk)
            A_t = torch.from_numpy(A_proc).to(device)
            pred, H, M = model(x, A_t)
            # unscale pred if labels were scaled
            pred_val = float(pred.detach().cpu().numpy()) * label_std + label_mean
            ys.append(float(item['y']) * label_std + label_mean if cfg.scale_labels else float(item['y']))
            preds.append(pred_val)
    if not ys:
        return {}
    ys = np.array(ys); preds = np.array(preds)
    rmse = math.sqrt(mean_squared_error(ys, preds))
    mae = mean_absolute_error(ys, preds)
    return {"rmse": rmse, "mae": mae, "n": len(ys)}

def train_one_epoch(model: MaskGNN, items: List[dict], optimizer, device: torch.device, cfg: argparse.Namespace) -> dict:
    """
    Returns dict with average total_loss, avg_task_loss, avg_l_man, avg_l_mask
    """
    model.train()
    total_loss = 0.0
    total_task = 0.0
    total_man = 0.0
    total_mask = 0.0
    count = 0
    for item in items:
        if item['y'] is None:
            continue
        x = torch.from_numpy(item['x']).to(device)
        A_proc = minmax_scale_adj(item['adj'])
        A_proc = keep_topk_per_row(A_proc, K=cfg.topk)
        A_t = torch.from_numpy(A_proc).to(device)
        # make target a scalar to match pred shape (pred is scalar tensor)
        y_t = torch.tensor(float(item['y']), dtype=torch.float32, device=device)  # shape: ()
        optimizer.zero_grad()
        pred, H, M = model(x, A_t)  # pred shape: ()
        task_loss = F.mse_loss(pred, y_t)  # shapes match now
        # manifold and mask penalties
        A_masked = (M + torch.eye(M.shape[0], device=M.device)) * model.normalize_adj(A_t)
        l_man = manifold_loss(H, A_masked)
        # normalize manifold by number of elements in H to keep magnitude comparable
        l_man = l_man / max(1.0, float(H.numel()))
        l_mask = mask_penalty(M, cfg.mask_l1, cfg.mask_l2, cfg.mask_orth)
        # normalize mask penalty by mask size
        l_mask = l_mask / max(1.0, float(M.numel()))
        loss = task_loss + cfg.alpha * l_man + l_mask
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().numpy())
        total_task += float(task_loss.detach().cpu().numpy())
        total_man += float(l_man.detach().cpu().numpy())
        total_mask += float(l_mask.detach().cpu().numpy())
        count += 1
    if count == 0:
        return {"loss": 0.0, "task": 0.0, "manifold": 0.0, "mask": 0.0}
    return {"loss": total_loss / count, "task": total_task / count, "manifold": total_man / count, "mask": total_mask / count}

# -----------------------------
# Label scaling helper
# -----------------------------
def fit_and_apply_label_scaler(train_items: List[dict], all_items: List[dict]):
    ys = np.array([it['y'] for it in train_items if it['y'] is not None], dtype=float)
    mean = float(np.mean(ys))
    std = float(np.std(ys)) if float(np.std(ys)) > 0 else 1.0
    print(f"Label scaler: mean={mean:.4f} std={std:.4f}")
    for it in all_items:
        if it['y'] is not None:
            it['y'] = (it['y'] - mean) / std
    return mean, std

# -----------------------------
# Feature scaling helper
# -----------------------------
def fit_and_apply_feature_scaler(train_items: List[dict], all_items: List[dict]):
    # train_items: list of dicts containing 'x' numpy arrays
    # fits MinMaxScaler per column on concatenated nodes across subjects
    # returns nothing but modifies items in-place
    print("Fitting MinMaxScaler across train node-features (per feature).")
    # stack across subjects and nodes
    X_stack = np.vstack([it['x'].reshape(-1, it['x'].shape[1]) for it in train_items])
    scaler = MinMaxScaler()
    scaler.fit(X_stack)
    # apply to all items
    for it in all_items:
        it['x'] = scaler.transform(it['x'])
    print("Applied feature scaler to all graphs.")

# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", type=str, default="graphs_prepared", help="folder with .pt graphs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--alpha", type=float, default=1.0, help="manifold regularization weight")
    p.add_argument("--mask-l1", type=float, default=1e-4)
    p.add_argument("--mask-l2", type=float, default=1e-4)
    p.add_argument("--mask-orth", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale-features", action="store_true", help="fit MinMax on train features and apply to all")
    p.add_argument("--scale-labels", action="store_true", help="z-score labels using train set")
    return p.parse_args()

def main():
    cfg = parse_args()
    device = get_device(prefer_cuda=(cfg.device.startswith("cuda")))
    print("Using device:", device)
    set_seed(cfg.seed)

    # load graphs
    items = load_graphs_from_dir(cfg.graphs)
    print(f"Loaded {len(items)} graphs from {cfg.graphs}")
    labeled = [it for it in items if it["y"] is not None]
    print(f"{len(labeled)} graphs have labels (will be used for supervised training).")
    if len(labeled) == 0:
        print("No labeled graphs found. Attach labels and retry.")
        return

    # split train/val
    random.shuffle(labeled)
    n = len(labeled)
    n_val = max(1, int(cfg.val_frac * n))
    val_items = labeled[:n_val]
    train_items = labeled[n_val:]
    print(f"Train count: {len(train_items)}, Val count: {len(val_items)}")

    # feature scaling if requested (fit on train)
    if cfg.scale_features:
        fit_and_apply_feature_scaler(train_items, items)

    # label scaling if requested
    label_mean, label_std = 0.0, 1.0
    if cfg.scale_labels:
        label_mean, label_std = fit_and_apply_label_scaler(train_items, items)

    Q = train_items[0]["x"].shape[0]
    in_dim = train_items[0]["x"].shape[1]
    print(f"Graph node count Q={Q}, input feature dim={in_dim}")

    model = MaskGNN(Q, in_dim, hidden_dims=[cfg.hidden_dim, cfg.hidden_dim], dropout=cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, cfg.epochs + 1):
        stats = train_one_epoch(model, train_items, optimizer, device, cfg)
        val_metrics = eval_model(model, val_items, device, cfg, label_mean=label_mean, label_std=label_std)
        print(f"[{epoch}/{cfg.epochs}] train_loss={stats['loss']:.6f} task={stats['task']:.6f} man={stats['manifold']:.6f} mask={stats['mask']:.6f} | val_rmse={val_metrics.get('rmse',None)} val_mae={val_metrics.get('mae',None)}")
        if val_metrics and val_metrics.get("rmse", float("inf")) < best_val:
            best_val = val_metrics["rmse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, "best_maskgnn.pt")
            with torch.no_grad():
                M = model.build_mask().cpu().numpy()
                np.save("best_mask.npy", M)
            print("  Saved new best model and mask (best_maskgnn.pt, best_mask.npy).")

    # final test-evaluation on val set if desired
    if best_state is not None:
        print("Loading best model for final evaluation.")
        model.load_state_dict(best_state)
    final_val = eval_model(model, val_items, device, cfg, label_mean=label_mean, label_std=label_std)
    print("Final val metrics:", final_val)
    print("Training complete. Saved best model/mask if validation improved.")

if __name__ == "__main__":
    main()
