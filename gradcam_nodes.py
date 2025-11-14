# gradcam_nodes.py (fixed & robust)
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# We reuse the MaskGNN class and helper functions from train.py
# to ensure identical architecture/processing. If train.py is in same folder, import it.
from train import MaskGNN, minmax_scale_adj, keep_topk_per_row

GRAPHS = "graphs_prepared"
MODEL_STATE = "best_maskgnn.pt"   # saved state_dict
OUTDIR = Path("gradcam_viz")
OUTDIR.mkdir(exist_ok=True)

# gather labeled graphs
graphs = []
for p in sorted(glob.glob(os.path.join(GRAPHS, "*.pt"))):
    g = torch.load(p)
    sid = Path(p).stem
    if isinstance(g, dict):
        x = g.get('x'); A = g.get('adj'); y = g.get('y')
    else:
        x = getattr(g, "x", None); A = getattr(g, "adj", None); y = getattr(g, "y", None)
    if y is None:
        continue
    graphs.append({'sid': sid, 'x': x, 'adj': A, 'y': float(y)})
if not graphs:
    raise SystemExit("No labeled graphs found in graphs_prepared/")

# Choose first labeled subject to explain
item = graphs[0]
print("Using subject:", item['sid'])

# Ensure x/A are numpy arrays or tensors handled uniformly
def to_tensor(x, device):
    # Accept torch.Tensor or np.ndarray
    if torch.is_tensor(x):
        t = x.to(device).float()
    else:
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    return t

Q = item['x'].shape[0] if not torch.is_tensor(item['x']) else item['x'].shape[0]
in_dim = item['x'].shape[1] if not torch.is_tensor(item['x']) else item['x'].shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "Q:", Q, "in_dim:", in_dim)

# build model
model = MaskGNN(Q, in_dim, hidden_dims=[64,64]).to(device)

# load saved state dict if available
if os.path.exists(MODEL_STATE):
    st = torch.load(MODEL_STATE, map_location=device)
    try:
        model.load_state_dict(st, strict=False)
        print("Loaded model state from", MODEL_STATE)
    except Exception as e:
        print("Warning: could not load state_dict fully:", e)
        # proceed with partial load / random init
else:
    print("No model state found at", MODEL_STATE, "; using random init.")

# Preprocess adjacency
A_proc = minmax_scale_adj(item['adj'])
A_proc = keep_topk_per_row(A_proc, K=30)
A_t = torch.from_numpy(A_proc).to(device).float()

# Prepare input tensor (make sure it's a tensor on device)
x_t = to_tensor(item['x'], device)

# We will capture the final node activations H from the last GNN layer using a forward hook
H_container = {"val": None}

def forward_hook(module, inp, out):
    # out is tensor with shape (Q, d)
    out.retain_grad()      # ensure gradients for this intermediate output are saved
    H_container["val"] = out

# register hook on the last MaskGNNLayer
last_layer = model.layers[-1]
hook_handle = last_layer.register_forward_hook(forward_hook)

# Forward pass
model.eval()
x_t.requires_grad = False  # we don't need gradients wrt input features themselves
pred, H, M = model(x_t, A_t)
# pred is scalar tensor
print("Model prediction:", float(pred.detach().cpu().numpy()))

# Backward: compute gradients of scalar prediction w.r.t H (captured by hook)
model.zero_grad()
if pred.requires_grad:
    pred.backward(retain_graph=True)
else:
    pred = pred.requires_grad_()
    pred.backward(retain_graph=True)

# Now H_container["val"] should hold the last-layer activations with grad populated
if H_container["val"] is None:
    # fallback: try to use H returned by forward (may not have grad)
    H_val = H.detach()
    if H.grad is None:
        raise SystemExit("Could not obtain activations grad; ensure model's last layer hook worked.")
    H_grad = H.grad.detach()
else:
    H_val = H_container["val"].detach()
    if H_container["val"].grad is None:
        # try to fetch grad from H (if it was set)
        # sometimes grad is present in H if retained
        try:
            H_grad = H_container["val"].grad.detach()
        except:
            # attempt to check H.grad
            H_grad = H.grad.detach() if H.grad is not None else None
    else:
        H_grad = H_container["val"].grad.detach()

if H_grad is None:
    raise SystemExit("Could not retrieve gradient of last-layer activations. Aborting.")

# compute Grad-CAM style node importance
H_np = H_val.cpu().numpy()    # shape (Q, d)
G_np = H_grad.cpu().numpy()   # shape (Q, d)
cam = (G_np * H_np).mean(axis=1)    # mean over feature dim
cam = np.maximum(cam, 0.0)    # ReLU
# normalize 0-1
if cam.max() > 0:
    cam = cam / (cam.max() + 1e-12)

# save node importances
out_npy = OUTDIR / f"{item['sid']}_node_importance.npy"
np.save(out_npy, cam)
print("Saved node importances to", out_npy)

# Simple circular plot for node importances
import matplotlib.pyplot as plt
import networkx as nx
Gplot = nx.path_graph(len(cam))  # dummy for layout
pos = nx.circular_layout(Gplot)
plt.figure(figsize=(8,8))
xs = [pos[i][0] for i in range(len(cam))]
ys = [pos[i][1] for i in range(len(cam))]
sizes = 50 + cam * 400
sc = plt.scatter(xs, ys, s=sizes, c=cam, cmap='hot', vmin=0, vmax=1)
plt.colorbar(sc, label="Node importance (Grad-CAM norm)")
plt.title(f"Node importances {item['sid']}")
plt.axis('off')
out_png = OUTDIR / f"{item['sid']}_node_importance.png"
plt.savefig(out_png, bbox_inches='tight', dpi=200)
print("Saved plot:", out_png)

# cleanup hook
hook_handle.remove()
