"""
Visualization script for pre-trained models on domain generalization datasets.
Loads a model checkpoint and evaluates on multiple domains, producing ONE image:
  - Large T-SNE Feature Space panel (color = class, shape = domain)
  - Inset accuracy bar chart (per-domain accuracy + avg line)
  - Legends for class colors and domain shapes
  - Title with per-domain accuracy and overall average

Usage:
  conda activate rw2s-vision
  python visualize_model.py \
    --model_path /home/adminn/theanh28/raven/w2s/pacs/models/train3_seed0.pt \
    --dataset pacs \
    --test_domains 0 1 3 \
    --device 0
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# ── Add project root to path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision.models import get_model


# =====================================================================
# Dataset configs
# =====================================================================
DATASET_INFO = {
    "pacs": {
        "domains": ["art_painting", "cartoon", "photo", "sketch"],
        "n_classes": 7,
        "class_names": ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
        "data_subdir": "pacs/images",
    },
    "vlcs": {
        "domains": ["CALTECH", "LABELME", "SUN", "PASCAL"],
        "n_classes": 5,
        "class_names": ["bird", "car", "chair", "dog", "person"],
        "data_subdir": "VLCS",
        "domain_suffix": "full",
    },
    "office_home": {
        "domains": ["art", "clipart", "product", "real_world"],
        "n_classes": 65,
        "class_names": None,  # too many, loaded from ImageFolder
        "data_subdir": "office_home_dg",
        "domain_suffix": "train",
    },
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =====================================================================
# Data loading
# =====================================================================
def get_domain_path(data_root, dataset_name, domain_name):
    info = DATASET_INFO[dataset_name]
    base = os.path.join(data_root, info["data_subdir"], domain_name)
    suffix = info.get("domain_suffix", None)
    if suffix:
        base = os.path.join(base, suffix)
    return base


def load_domain_data(data_root, dataset_name, domain_name, batch_size=64, num_workers=4):
    path = get_domain_path(data_root, dataset_name, domain_name)
    dataset = ImageFolder(root=path, transform=TRANSFORM)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataset, loader


# =====================================================================
# Model loading
# =====================================================================
def load_model_from_checkpoint(model_path, device_id=0):
    """Load model from checkpoint, return model and metadata."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    cfg = ckpt["cfg"]
    model_cfg = ckpt["model_cfg"]
    state_dict = ckpt["state_dict"]

    model_name = model_cfg["model_name"]
    dataset_name = cfg["data"]["name"]
    n_classes = DATASET_INFO[dataset_name]["n_classes"]

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    model = get_model(
        name=model_name,
        device=device,
        pretrained=False,
        replace_last_layer_with_n_classes=n_classes,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, model_name, dataset_name, cfg, device


# =====================================================================
# Evaluation
# =====================================================================
@torch.no_grad()
def evaluate_on_domain(model, loader, device):
    """
    Returns:
        embeddings: (N, D)  latent vectors
        probs:      (N, C)  softmax outputs
        preds:      (N,)    predicted classes
        labels:     (N,)    ground-truth classes
        acc:        float   accuracy
    """
    all_embs, all_probs, all_labels = [], [], []
    for imgs, targets in tqdm(loader, desc="  Evaluating", leave=False):
        imgs = imgs.to(device)
        out = model(imgs)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            embs, logits = out
        else:
            logits = out
            embs = logits  # fallback
        probs = torch.softmax(logits, dim=-1)
        all_embs.append(embs.cpu())
        all_probs.append(probs.cpu())
        all_labels.append(targets)

    embeddings = torch.cat(all_embs, dim=0).numpy()
    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = probs.argmax(axis=1)
    acc = (preds == labels).mean() * 100.0
    return embeddings, probs, preds, labels, acc


# =====================================================================
# Plotting — single image output
# =====================================================================
def plot_all(results, dataset_name, model_name, save_dir):
    """
    results: list of dicts with keys:
        domain_name, embeddings, probs, preds, labels, acc

    Produces ONE single PNG:
      - Large T-SNE Feature Space  (color = class, shape = domain)
      - Decision boundary overlay
      - Inset horizontal bar chart: per-domain accuracy + avg line
      - Two legends: domain shapes (upper-left), class colors (lower-right)
      - Title includes per-domain accuracy and average
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE
    from sklearn.linear_model import LogisticRegression

    info = DATASET_INFO[dataset_name]
    n_classes = info["n_classes"]
    domain_names = [r["domain_name"] for r in results]
    n_domains = len(results)
    accs = [r["acc"] for r in results]
    avg_acc = float(np.mean(accs))

    # ── Global style ──
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    # Class colors
    if n_classes <= 10:
        cmap_cls = plt.colormaps.get_cmap("tab10").resampled(10)
    else:
        cmap_cls = plt.colormaps.get_cmap("tab20").resampled(20)
    class_colors = [cmap_cls(i % cmap_cls.N) for i in range(n_classes)]

    # Domain markers (distinct shapes)
    MARKER_POOL = ["o", "s", "^", "D", "v", "P", "*", "X"]
    domain_markers = MARKER_POOL[:n_domains]

    # Domain bar colors
    BAR_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373",
                  "#ba68c8", "#4db6ac", "#f06292", "#a1887f"]
    bar_colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(n_domains)]

    # ================================================================
    # 1. T-SNE on merged dataset D = union of all test domains
    # ================================================================
    print("  Computing T-SNE on merged dataset D (may take a while)...")

    all_embs = np.concatenate([r["embeddings"] for r in results], axis=0)
    all_labels = np.concatenate([r["labels"] for r in results], axis=0)
    all_preds = np.concatenate([r["preds"] for r in results], axis=0)
    all_domain_ids = np.concatenate(
        [np.full(len(r["labels"]), i) for i, r in enumerate(results)], axis=0
    )

    max_points = 5000
    if len(all_embs) > max_points:
        rng = np.random.RandomState(42)
        sel = rng.choice(len(all_embs), max_points, replace=False)
        all_embs = all_embs[sel]
        all_labels = all_labels[sel]
        all_preds = all_preds[sel]
        all_domain_ids = all_domain_ids[sel]
        print(f"  Subsampled to {max_points} points for T-SNE")

    perp = min(30, max(5, len(all_embs) - 1))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=1000)
    X_2d = tsne.fit_transform(all_embs)

    # Surrogate decision boundary (logistic regression in 2D tsne space)
    lr = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    lr.fit(X_2d, all_preds)

    margin = 0.08
    rx = X_2d[:, 0].max() - X_2d[:, 0].min()
    ry = X_2d[:, 1].max() - X_2d[:, 1].min()
    x_min = X_2d[:, 0].min() - margin * rx
    x_max = X_2d[:, 0].max() + margin * rx
    y_min = X_2d[:, 1].min() - margin * ry
    y_max = X_2d[:, 1].max() + margin * ry
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    Z_pred = lr.predict(grid_pts).reshape(xx.shape)
    Z_conf = lr.predict_proba(grid_pts).max(axis=1).reshape(xx.shape)

    # ================================================================
    # 2. Build single figure
    #    - Main area  [left=0.05, bottom=0.08, w=0.70, h=0.84]: T-SNE
    #    - Inset area [left=0.78, bottom=0.55, w=0.20, h=0.35]: acc bar
    # ================================================================
    fig = plt.figure(figsize=(17, 11), facecolor="#f4f6f8")

    ax_tsne = fig.add_axes([0.04, 0.07, 0.70, 0.85])
    ax_bar  = fig.add_axes([0.77, 0.56, 0.21, 0.34])

    # ── Decision boundary overlay ──
    # ax_tsne.contourf(xx, yy, Z_conf, levels=20, cmap="Greys", alpha=0.18)
    # ax_tsne.contour(xx, yy, Z_pred, colors="gray", linewidths=0.5, alpha=0.40)

    # ── Scatter: shape = domain, color = class ──
    unique_classes = np.unique(all_labels)
    for di in range(n_domains):
        d_mask = all_domain_ids == di
        for c in unique_classes:
            mask = d_mask & (all_labels == c)
            if mask.sum() == 0:
                continue
            ax_tsne.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                color=class_colors[int(c) % len(class_colors)],
                marker=domain_markers[di],
                s=32, alpha=0.78,
                edgecolors="white", linewidths=0.3,
            )

    ax_tsne.set_xlabel("T-SNE Dimension 1", fontsize=12)
    ax_tsne.set_ylabel("T-SNE Dimension 2", fontsize=12)
    ax_tsne.set_facecolor("#ffffff")
    ax_tsne.set_xlim(x_min, x_max)
    ax_tsne.set_ylim(y_min, y_max)
    for sp in ["top", "right"]:
        ax_tsne.spines[sp].set_visible(False)

    # Title with per-domain accuracies
    acc_parts = "   ".join([f"{dn}: {ac:.1f}%" for dn, ac in zip(domain_names, accs)])
    ax_tsne.set_title(
        f"Feature Space Visualization — {model_name}  |  {dataset_name.upper()}\n"
        f"Accuracy  →  {acc_parts}   |   Avg: {avg_acc:.1f}%",
        fontsize=13, fontweight="bold", pad=14,
    )

    # Legend 1: domain shapes → upper-left
    domain_leg_handles = [
        Line2D([0], [0], marker=domain_markers[di], color="w",
               markerfacecolor="#555555", markeredgecolor="#222222",
               markersize=11, label=f"{dn}  ({accs[di]:.1f}%)")
        for di, dn in enumerate(domain_names)
    ]
    leg_dom = ax_tsne.legend(
        handles=domain_leg_handles,
        loc="upper left",
        title="Domain  (shape)", title_fontsize=9,
        framealpha=0.88, edgecolor="#cccccc",
    )
    ax_tsne.add_artist(leg_dom)

    # Legend 2: class colors → lower-right
    show_n = min(20, len(unique_classes))
    class_leg_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=class_colors[int(c) % len(class_colors)],
               markersize=9,
               label=info["class_names"][int(c)] if info["class_names"] else f"Class {int(c)}")
        for c in unique_classes[:show_n]
    ]
    if len(unique_classes) > show_n:
        class_leg_handles.append(
            Line2D([0], [0], marker="", color="w",
                   label=f"... +{len(unique_classes) - show_n} more")
        )
    ncol_cls = 1 if n_classes <= 10 else 2
    ax_tsne.legend(
        handles=class_leg_handles,
        loc="lower right",
        title="Class  (color)", title_fontsize=9,
        framealpha=0.88, edgecolor="#cccccc", ncol=ncol_cls,
    )

    # ── Inset accuracy bar chart ──
    bars = ax_bar.barh(
        domain_names, accs,
        color=bar_colors, edgecolor="white", linewidth=1.2, height=0.55,
    )
    for bar, ac in zip(bars, accs):
        label_x = max(ac - 1.5, 0.5)
        ax_bar.text(
            label_x, bar.get_y() + bar.get_height() / 2,
            f"{ac:.1f}%", va="center", ha="right",
            fontsize=8, fontweight="bold", color="white",
        )
    ax_bar.set_xlim(0, max(accs) + 14)
    ax_bar.set_xlabel("Accuracy (%)", fontsize=8)
    ax_bar.set_title("Per-Domain Accuracy", fontsize=9, fontweight="bold")
    ax_bar.axvline(avg_acc, color="#e53935", ls="--", lw=1.3,
                   label=f"Avg  {avg_acc:.1f}%")
    ax_bar.legend(fontsize=7, framealpha=0.75, loc="lower right")
    ax_bar.tick_params(axis="y", labelsize=8)
    ax_bar.tick_params(axis="x", labelsize=7)
    ax_bar.set_facecolor("#ffffff")
    ax_bar.grid(axis="x", alpha=0.25)
    for sp in ["top", "right"]:
        ax_bar.spines[sp].set_visible(False)

    # ── Save ──
    out_path = os.path.join(save_dir, "feature_space_visualization.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Saved] {out_path}")

    # ── Summary text ──
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Domains evaluated: {domain_names}\n")
        f.write("=" * 50 + "\n")
        for r in results:
            f.write(f"\n  Domain: {r['domain_name']}\n")
            f.write(f"    Accuracy : {r['acc']:.2f}%\n")
            f.write(f"    N samples: {len(r['labels'])}\n")
            mp = r["probs"].max(axis=1)
            f.write(f"    Mean max softmax: {mp.mean():.4f}\n")
            ent = -(r["probs"] * np.log(r["probs"] + 1e-8)).sum(axis=1)
            f.write(f"    Mean entropy    : {ent.mean():.4f}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Average accuracy: {avg_acc:.2f}%\n")
    print(f"  [Saved] {summary_path}")


# =====================================================================
# Plotting — Confidence KDE (Image 2)
# =====================================================================
def plot_confidence_kde(results, dataset_name, model_name, save_dir):
    """
    Image 2 — Confidence Density (KDE) plots.
    One subplot per model (here: one figure panel per model checkpoint).
    Draws 4 KDE curves:
      - d1, d2, d3 : individual test-domain confidence distributions
      - D          : merged (d1 ∪ d2 ∪ d3) confidence distribution
    X-axis: max-softmax confidence [0, 1]
    Y-axis: density
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9.5,
        "figure.dpi": 150,
    })

    n_domains = len(results)
    domain_names = [r["domain_name"] for r in results]
    accs = [r["acc"] for r in results]
    avg_acc = float(np.mean(accs))

    # Per-domain max-softmax confidence
    conf_per_domain = [r["probs"].max(axis=1) for r in results]
    # Merged D
    conf_D = np.concatenate(conf_per_domain, axis=0)

    # Color palette: one color per domain + one for D
    DOMAIN_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373",
                     "#ba68c8", "#4db6ac", "#f06292", "#a1887f"]
    D_COLOR = "#212121"

    x_grid = np.linspace(0, 1, 500)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#f4f6f8")
    ax.set_facecolor("#ffffff")

    # Draw KDE for each di
    for i, (conf, dn, acc) in enumerate(zip(conf_per_domain, domain_names, accs)):
        color = DOMAIN_COLORS[i % len(DOMAIN_COLORS)]
        if len(conf) > 1:
            kde = gaussian_kde(conf, bw_method="scott")
            ax.plot(x_grid, kde(x_grid),
                    color=color, linewidth=2.2,
                    label=f"d{i+1} — {dn}  (acc {acc:.1f}%)")
            ax.fill_between(x_grid, kde(x_grid), alpha=0.12, color=color)

    # Draw KDE for D (merged)
    if len(conf_D) > 1:
        kde_D = gaussian_kde(conf_D, bw_method="scott")
        ax.plot(x_grid, kde_D(x_grid),
                color=D_COLOR, linewidth=2.5, linestyle="--",
                label=f"D = {'∪'.join([f'd{i+1}' for i in range(n_domains)])}  (avg acc {avg_acc:.1f}%)")
        ax.fill_between(x_grid, kde_D(x_grid), alpha=0.08, color=D_COLOR)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence (max softmax)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Confidence Density (KDE) — {model_name}  |  {dataset_name.upper()}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(loc="upper left", framealpha=0.88, edgecolor="#cccccc")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    out_path = os.path.join(save_dir, "confidence_kde.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# =====================================================================
# Plotting — t-SNE on Logit Space (Image 3)
# =====================================================================
def plot_tsne_logit_space(results, dataset_name, model_name, save_dir):
    """
    Image 3 — t-SNE on Logit Space.
    Runs t-SNE on the raw logit vectors (log-softmax outputs) of merged D.
    Main plot  : domain = marker shape, class = color
    Inset plot : domain = color only (no class distinction)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from sklearn.manifold import TSNE

    info = DATASET_INFO[dataset_name]
    n_classes = info["n_classes"]
    domain_names = [r["domain_name"] for r in results]
    n_domains = len(results)
    accs = [r["acc"] for r in results]
    avg_acc = float(np.mean(accs))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    # ── Prepare logit vectors (log-softmax of probs) ──
    # probs are already softmax; logit approx = log(p + eps)
    all_logits = np.concatenate(
        [np.log(r["probs"] + 1e-8) for r in results], axis=0
    )
    all_labels = np.concatenate([r["labels"] for r in results], axis=0)
    all_domain_ids = np.concatenate(
        [np.full(len(r["labels"]), i) for i, r in enumerate(results)], axis=0
    )

    # Subsample for speed
    max_points = 5000
    if len(all_logits) > max_points:
        rng = np.random.RandomState(42)
        sel = rng.choice(len(all_logits), max_points, replace=False)
        all_logits     = all_logits[sel]
        all_labels     = all_labels[sel]
        all_domain_ids = all_domain_ids[sel]
        print(f"  [t-SNE logit] Subsampled to {max_points} points")

    print("  Computing t-SNE on Logit Space (may take a while)...")
    perp = min(30, max(5, len(all_logits) - 1))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=1000)
    X_2d = tsne.fit_transform(all_logits)

    # ── Color / marker palettes ──
    if n_classes <= 10:
        cmap_cls = plt.colormaps.get_cmap("tab10").resampled(10)
    else:
        cmap_cls = plt.colormaps.get_cmap("tab20").resampled(20)
    class_colors = [cmap_cls(i % cmap_cls.N) for i in range(n_classes)]

    MARKER_POOL = ["o", "s", "^", "D", "v", "P", "*", "X"]
    domain_markers = MARKER_POOL[:n_domains]

    DOMAIN_COLORS_INSET = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373",
                           "#ba68c8", "#4db6ac", "#f06292", "#a1887f"]

    # ── Axis limits ──
    margin = 0.08
    rx = X_2d[:, 0].max() - X_2d[:, 0].min()
    ry = X_2d[:, 1].max() - X_2d[:, 1].min()
    x_min = X_2d[:, 0].min() - margin * rx
    x_max = X_2d[:, 0].max() + margin * rx
    y_min = X_2d[:, 1].min() - margin * ry
    y_max = X_2d[:, 1].max() + margin * ry

    # ── Figure ──
    fig = plt.figure(figsize=(14, 9), facecolor="#f4f6f8")
    ax_main = fig.add_axes([0.05, 0.07, 0.68, 0.86])

    # Main scatter: shape = domain, color = class
    unique_classes = np.unique(all_labels)
    for di in range(n_domains):
        d_mask = all_domain_ids == di
        for c in unique_classes:
            mask = d_mask & (all_labels == c)
            if mask.sum() == 0:
                continue
            ax_main.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                color=class_colors[int(c) % len(class_colors)],
                marker=domain_markers[di],
                s=30, alpha=0.78,
                edgecolors="white", linewidths=0.3,
            )

    ax_main.set_xlabel("t-SNE Dim 1", fontsize=12)
    ax_main.set_ylabel("t-SNE Dim 2", fontsize=12)
    ax_main.set_facecolor("#ffffff")
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    for sp in ["top", "right"]:
        ax_main.spines[sp].set_visible(False)

    acc_parts = "   ".join([f"{dn}: {ac:.1f}%" for dn, ac in zip(domain_names, accs)])
    ax_main.set_title(
        f"t-SNE on Logit Space — {model_name}  |  {dataset_name.upper()}\n"
        f"domain = shape  ·  class = color   |   Acc → {acc_parts}   Avg {avg_acc:.1f}%",
        fontsize=12, fontweight="bold", pad=12,
    )

    # Legend — domain shapes
    domain_leg = [
        Line2D([0], [0], marker=domain_markers[di], color="w",
               markerfacecolor="#555", markeredgecolor="#222",
               markersize=10, label=f"d{di+1} — {dn}  ({accs[di]:.1f}%)")
        for di, dn in enumerate(domain_names)
    ]
    leg1 = ax_main.legend(handles=domain_leg, loc="upper left",
                          title="Domain (shape)", title_fontsize=9,
                          framealpha=0.88, edgecolor="#ccc")
    ax_main.add_artist(leg1)

    # Legend — class colors
    show_n = min(20, len(unique_classes))
    cls_leg = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=class_colors[int(c) % len(class_colors)],
               markersize=9,
               label=info["class_names"][int(c)] if info["class_names"] else f"Class {int(c)}")
        for c in unique_classes[:show_n]
    ]
    if len(unique_classes) > show_n:
        cls_leg.append(Line2D([0], [0], marker="", color="w",
                               label=f"... +{len(unique_classes) - show_n} more"))
    ncol = 1 if n_classes <= 10 else 2
    ax_main.legend(handles=cls_leg, loc="lower right",
                   title="Class (color)", title_fontsize=9,
                   framealpha=0.88, edgecolor="#ccc", ncol=ncol)

    # ── Inset: domain coloring only ──
    ax_inset = fig.add_axes([0.76, 0.10, 0.22, 0.38])
    for di in range(n_domains):
        d_mask = all_domain_ids == di
        ax_inset.scatter(
            X_2d[d_mask, 0], X_2d[d_mask, 1],
            color=DOMAIN_COLORS_INSET[di % len(DOMAIN_COLORS_INSET)],
            marker="o", s=8, alpha=0.55,
            label=f"d{di+1} — {domain_names[di]}",
        )
    ax_inset.set_xlim(x_min, x_max)
    ax_inset.set_ylim(y_min, y_max)
    ax_inset.set_facecolor("#ffffff")
    ax_inset.set_title("Domain only (color)", fontsize=8, fontweight="bold", pad=4)
    ax_inset.tick_params(labelsize=6)
    ax_inset.legend(fontsize=6, framealpha=0.8, edgecolor="#ccc",
                    loc="upper left", markerscale=1.4)
    for sp in ["top", "right"]:
        ax_inset.spines[sp].set_visible(False)

    out_path = os.path.join(save_dir, "tsne_logit_space.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize a pre-trained model on domain generalization datasets"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["pacs", "vlcs", "office_home"],
                        help="Dataset name (auto-detected from checkpoint if not given)")
    parser.add_argument("--test_domains", type=int, nargs="+", default=None,
                        help="Domain indices to test on (e.g. 0 1 3). "
                             "Default: all except teacher_domain from checkpoint config")
    parser.add_argument("--data_root", type=str, default="/home/adminn/theanh28/DATA",
                        help="Root directory for datasets")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model_path}")
    model, model_name, dataset_name, cfg, device = load_model_from_checkpoint(
        args.model_path, args.device
    )

    if args.dataset is not None:
        dataset_name = args.dataset
    print(f"  Model: {model_name}, Dataset: {dataset_name}")

    info = DATASET_INFO[dataset_name]
    all_domains = info["domains"]

    # Determine which domains to test
    if args.test_domains is not None:
        test_domain_indices = args.test_domains
    else:
        teacher_idx = cfg["data"].get("teacher_domain", None)
        test_domain_indices = [i for i in range(len(all_domains)) if i != teacher_idx]

    test_domain_names = [all_domains[i] for i in test_domain_indices]
    print(f"  Test domains: {test_domain_names}")

    # Save directory
    if args.save_dir is None:
        ckpt_basename = os.path.splitext(os.path.basename(args.model_path))[0]
        save_dir = os.path.join(
            os.path.dirname(args.model_path), "plots",
            f"{ckpt_basename}__{dataset_name}__{'_'.join(test_domain_names)}"
        )
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"  Save dir: {save_dir}")

    # Evaluate each domain
    results = []
    for domain_name in test_domain_names:
        print(f"\n  Evaluating on domain: {domain_name}")
        _, loader = load_domain_data(
            args.data_root, dataset_name, domain_name,
            batch_size=args.batch_size, num_workers=4,
        )
        embeddings, probs, preds, labels, acc = evaluate_on_domain(model, loader, device)
        print(f"    Accuracy: {acc:.2f}%  |  N={len(labels)}")
        results.append({
            "domain_name": domain_name,
            "embeddings": embeddings,
            "probs": probs,
            "preds": preds,
            "labels": labels,
            "acc": acc,
        })

    # Generate the single combined visualization
    print("\nGenerating feature space visualization...")
    plot_all(results, dataset_name, model_name, save_dir)

    # Ảnh 2: Confidence KDE
    print("\nGenerating confidence KDE plot...")
    plot_confidence_kde(results, dataset_name, model_name, save_dir)

    # Ảnh 3: t-SNE on Logit Space
    print("\nGenerating t-SNE on Logit Space plot...")
    plot_tsne_logit_space(results, dataset_name, model_name, save_dir)

    print(f"\nDone! Output saved to: {save_dir}")


if __name__ == "__main__":
    main()
