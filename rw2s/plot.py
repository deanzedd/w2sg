import os
import numpy as np
import torch
# =============================================================================
# W2SG Analysis Visualization (toggleable via cfg["w2s"]["plot_w2sg"])
# =============================================================================
def plot_w2sg_analysis(
    model,
    eval_datasets,
    save_dir,
    seed,
    device,
    dim_reduction='pca',
    gt_model=None,
    balance_stats=None,
    logger=None,
):
    """
    Generate academic-grade visualization plots for Weak-to-Strong Generalization
    (W2SG) analysis on the test set. Produces a figure with 6 subplots:

      1.  Weak Model Decision Space  (contour + scatter by GT label)
      2a. Strong Model (W2SG, trained on weak labels) Decision Space
      2b. GT Model (trained on ground truth) Decision Space
      3.  Strong Model + 4 Emphasized Data Groups
      4.  Entropy KDE  (Weak vs Strong density overlay)
      5.  Weak Entropy vs Strong Entropy Scatter (colored by 4 groups)

    A summary panel shows textual group statistics.

    Args:
        model:          Trained w2sg linear probe (nn.Linear or similar).
        eval_datasets:  dict containing at least 'test', 'test_weak',
                        and optionally 'test_weak_raw'.
        save_dir:       Directory to save the output PNG.
        seed:           Seed number (used in the filename).
        device:         torch device.
        dim_reduction:  'pca' or 'tsne' for 2-D projection.
        gt_model:       Trained GT linear probe (nn.Linear or similar),
                        trained on ground-truth labels. If None, Plot 2b
                        is skipped.
        logger:         Optional logger instance.
    """
    # ---- lazy imports (avoid breaking existing code if libs missing) ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    os.makedirs(save_dir, exist_ok=True)

    # ==================================================================
    # 1. Extract test data from eval_datasets
    # ==================================================================
    x_test, y_test = eval_datasets["test"]
    _, yw_test = eval_datasets["test_weak"]

    x_test_tensor = x_test.float().to(device)

    # ==================================================================
    # 2. Strong model predictions + entropy
    # ==================================================================
    model.eval()
    with torch.no_grad():
        logits_strong = model(x_test_tensor).detach().cpu()
        if logits_strong.ndim > 2:
            logits_strong = logits_strong.mean(1)
        probs_strong = torch.softmax(logits_strong, dim=-1)
        pred_strong  = torch.argmax(logits_strong, dim=-1)
        entropy_strong = -(probs_strong * torch.log(probs_strong + 1e-8)).sum(dim=-1)

    # ==================================================================
    # 2b. GT model predictions (if gt_model is provided)
    # ==================================================================
    has_gt = gt_model is not None
    if has_gt:
        gt_model.eval()
        with torch.no_grad():
            logits_gt = gt_model(x_test_tensor).detach().cpu()
            if logits_gt.ndim > 2:
                logits_gt = logits_gt.mean(1)
            pred_gt = torch.argmax(logits_gt, dim=-1)

    # ==================================================================
    # 3. Weak predictions + entropy
    # ==================================================================
    y_true = y_test.argmax(-1).cpu() if y_test.ndim > 1 else y_test.cpu()
    y_weak = yw_test.argmax(-1).cpu() if yw_test.ndim > 1 else yw_test.cpu()

    weak_raw_available = "test_weak_raw" in eval_datasets
    if weak_raw_available:
        _, yw_test_raw = eval_datasets["test_weak_raw"]
        if isinstance(yw_test_raw, np.ndarray):
            yw_test_raw = torch.tensor(yw_test_raw)
        yw_test_raw = yw_test_raw.float()
        if yw_test_raw.min() >= 0 and yw_test_raw.max() <= 1.01:
            weak_probs = yw_test_raw
        else:
            weak_probs = torch.softmax(yw_test_raw, dim=-1)
        entropy_weak = -(weak_probs * torch.log(weak_probs + 1e-8)).sum(dim=-1).cpu()
    else:
        entropy_weak = torch.zeros(len(y_true))

    # ---- convert to numpy ----
    y_true_np       = y_true.numpy()
    y_weak_np       = y_weak.numpy()
    y_strong_np     = pred_strong.numpy()
    y_gt_np         = pred_gt.numpy() if has_gt else None
    entropy_weak_np = entropy_weak.numpy()
    entropy_strong_np = entropy_strong.numpy()
    x_test_np = x_test.cpu().numpy() if isinstance(x_test, torch.Tensor) else np.array(x_test)

    # ==================================================================
    # 4. 2-D projection
    # ==================================================================
    if dim_reduction == 'tsne':
        from sklearn.manifold import TSNE
        perp = min(30, max(5, len(x_test_np) - 1))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perp)
        X_2d = reducer.fit_transform(x_test_np)
    else:  # default: pca
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(x_test_np)

    # ==================================================================
    # 5. Define 4 logical groups
    # ==================================================================
    weak_correct   = (y_weak_np == y_true_np)
    strong_correct = (y_strong_np == y_true_np)

    group_A = weak_correct  & strong_correct    # Both correct  (easy)
    group_B = ~weak_correct & strong_correct     # W2SG phenomenon ★
    group_C = weak_correct  & ~strong_correct    # Negative transfer
    group_D = ~weak_correct & ~strong_correct    # Both wrong (hard/noise)

    # ==================================================================
    # 6. Fit surrogate logistic regressions in 2-D for contour plots
    # ==================================================================
    lr_weak = LogisticRegression(max_iter=1000, multi_class='multinomial',
                                  solver='lbfgs', random_state=42)
    lr_weak.fit(X_2d, y_weak_np)

    lr_strong = LogisticRegression(max_iter=1000, multi_class='multinomial',
                                    solver='lbfgs', random_state=42)
    lr_strong.fit(X_2d, y_strong_np)

    # GT surrogate (if gt_model provided)
    if has_gt:
        lr_gt = LogisticRegression(max_iter=1000, multi_class='multinomial',
                                    solver='lbfgs', random_state=42)
        lr_gt.fit(X_2d, y_gt_np)

    # ---- mesh grid ----
    margin = 0.08
    rx = X_2d[:, 0].max() - X_2d[:, 0].min()
    ry = X_2d[:, 1].max() - X_2d[:, 1].min()
    x_min, x_max = X_2d[:, 0].min() - margin * rx, X_2d[:, 0].max() + margin * rx
    y_min, y_max = X_2d[:, 1].min() - margin * ry, X_2d[:, 1].max() + margin * ry
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]

    Z_weak   = lr_weak.predict(grid_pts).reshape(xx.shape)
    Z_strong = lr_strong.predict(grid_pts).reshape(xx.shape)
    Z_weak_conf   = lr_weak.predict_proba(grid_pts).max(axis=1).reshape(xx.shape)
    Z_strong_conf = lr_strong.predict_proba(grid_pts).max(axis=1).reshape(xx.shape)

    if has_gt:
        Z_gt      = lr_gt.predict(grid_pts).reshape(xx.shape)
        Z_gt_conf = lr_gt.predict_proba(grid_pts).max(axis=1).reshape(xx.shape)

    # ==================================================================
    # 7. Aesthetic setup
    # ==================================================================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
    })

    unique_classes = np.unique(y_true_np)
    n_cls = len(unique_classes)
    cmap_cls = plt.cm.get_cmap('tab10', max(n_cls, 10))
    dim_label = dim_reduction.upper()

    # ==================================================================
    # 8. Create figure  (2 × 4 grid)
    # ==================================================================
    fig = plt.figure(figsize=(28, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.30)

    # ---------- PLOT 1: Weak Model Decision Space ----------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.contourf(xx, yy, Z_weak_conf, levels=20, cmap='Blues', alpha=0.35)
    ax1.contour(xx, yy, Z_weak, colors='navy', linewidths=0.7, alpha=0.55)
    for ci, c in enumerate(unique_classes):
        mask_c = y_true_np == c
        mrk = 'o' if ci % 2 == 0 else 'x'
        ax1.scatter(X_2d[mask_c, 0], X_2d[mask_c, 1],
                    c=[cmap_cls(ci)], marker=mrk, s=18, alpha=0.7,
                    edgecolors='white' if mrk == 'o' else 'none',
                    linewidths=0.3, label=f'Class {int(c)}')
    ax1.set_title('Plot 1: Weak Model Decision Space', fontweight='bold')
    ax1.set_xlabel(f'{dim_label} Dim 1'); ax1.set_ylabel(f'{dim_label} Dim 2')
    ax1.legend(loc='best', framealpha=0.85, markerscale=1.3)

    # ---------- PLOT 2a: Strong Model (W2SG) Decision Space ----------
    ax2a = fig.add_subplot(gs[0, 1])
    ax2a.contourf(xx, yy, Z_strong_conf, levels=20, cmap='Oranges', alpha=0.35)
    ax2a.contour(xx, yy, Z_strong, colors='darkred', linewidths=0.7, alpha=0.55)
    for ci, c in enumerate(unique_classes):
        mask_c = y_true_np == c
        mrk = 'o' if ci % 2 == 0 else 'x'
        ax2a.scatter(X_2d[mask_c, 0], X_2d[mask_c, 1],
                    c=[cmap_cls(ci)], marker=mrk, s=18, alpha=0.7,
                    edgecolors='white' if mrk == 'o' else 'none',
                    linewidths=0.3, label=f'Class {int(c)}')
    ax2a.set_title('Plot 2a: Strong (W2SG, Weak Labels)', fontweight='bold')
    ax2a.set_xlabel(f'{dim_label} Dim 1'); ax2a.set_ylabel(f'{dim_label} Dim 2')
    ax2a.legend(loc='best', framealpha=0.85, markerscale=1.3)

    # ---------- PLOT 2b: GT Model Decision Space ----------
    ax2b = fig.add_subplot(gs[0, 2])
    if has_gt:
        ax2b.contourf(xx, yy, Z_gt_conf, levels=20, cmap='Greens', alpha=0.35)
        ax2b.contour(xx, yy, Z_gt, colors='darkgreen', linewidths=0.7, alpha=0.55)
        for ci, c in enumerate(unique_classes):
            mask_c = y_true_np == c
            mrk = 'o' if ci % 2 == 0 else 'x'
            ax2b.scatter(X_2d[mask_c, 0], X_2d[mask_c, 1],
                        c=[cmap_cls(ci)], marker=mrk, s=18, alpha=0.7,
                        edgecolors='white' if mrk == 'o' else 'none',
                        linewidths=0.3, label=f'Class {int(c)}')
        gt_acc = (y_gt_np == y_true_np).sum() / len(y_true_np) * 100
        ax2b.set_title(f'Plot 2b: Strong (GT Labels) Acc={gt_acc:.1f}%', fontweight='bold')
    else:
        ax2b.text(0.5, 0.5, 'GT model not provided', ha='center', va='center',
                  transform=ax2b.transAxes, fontsize=12, color='gray')
        ax2b.set_title('Plot 2b: Strong (GT Labels) [N/A]', fontweight='bold')
    ax2b.set_xlabel(f'{dim_label} Dim 1'); ax2b.set_ylabel(f'{dim_label} Dim 2')
    if has_gt:
        ax2b.legend(loc='best', framealpha=0.85, markerscale=1.3)

    # ---------- PLOT 3: Strong Model + 4 Groups ----------
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.contourf(xx, yy, Z_strong_conf, levels=20, cmap='Greys', alpha=0.2)
    ax3.contour(xx, yy, Z_strong, colors='gray', linewidths=0.5, alpha=0.35)

    group_specs = [
        (group_A, '#2ca02c', 'o',  'A: Both Correct (Easy)',       22),
        (group_B, '#ff7f0e', '*',  'B: W2SG Phenomenon \u2605',    65),
        (group_C, '#d62728', '^',  'C: Negative Transfer',         28),
        (group_D, '#7f7f7f', 'x',  'D: Both Wrong (Hard)',         22),
    ]
    for mask, color, mrk, label, sz in group_specs:
        if mask.sum() > 0:
            ax3.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, marker=mrk,
                        s=sz, alpha=0.8,
                        label=f'{label} ({mask.sum()})',
                        edgecolors='black' if mrk in ('o', '^', '*') else 'none',
                        linewidths=0.3)
    ax3.set_title('Plot 3: Strong Model + 4 Data Groups', fontweight='bold')
    ax3.set_xlabel(f'{dim_label} Dim 1'); ax3.set_ylabel(f'{dim_label} Dim 2')
    ax3.legend(loc='best', framealpha=0.85, fontsize=7, markerscale=1.0)

    # ---------- PLOT (Balance Stats): Class Balance Before/After Filtering ----------
    ax_bal = fig.add_subplot(gs[1, 0])
    ax_bal.axis('off')
    if balance_stats is not None:
        bs_before = balance_stats.get('before', {})
        bs_after  = balance_stats.get('after', {})
        ir_b = f"{bs_before.get('imbalance_ratio', 0):.2f}" if bs_before.get('imbalance_ratio', 0) != float('inf') else "inf"
        ir_a = f"{bs_after.get('imbalance_ratio', 0):.2f}" if bs_after.get('imbalance_ratio', 0) != float('inf') else "inf"
        bal_text = (
            f"Class Balance Analysis\n"
            f"{'=' * 40}\n\n"
            f"BEFORE Filtering:\n"
            f"  Samples:          {bs_before.get('n_total', '?')}\n"
            f"  Zero-sample cls:  {bs_before.get('zero_sample_classes', '?')}\n"
            f"  Zero cls IDs:     {str(bs_before.get('zero_class_ids', [])[:8])}\n"
            f"  Imbalance Ratio:  {ir_b}\n"
            f"  Balance Entropy:  {bs_before.get('balance_entropy', 0):.4f}\n\n"
            f"AFTER Filtering:\n"
            f"  Samples:          {bs_after.get('n_total', '?')}\n"
            f"  Zero-sample cls:  {bs_after.get('zero_sample_classes', '?')}\n"
            f"  Zero cls IDs:     {str(bs_after.get('zero_class_ids', [])[:8])}\n"
            f"  Imbalance Ratio:  {ir_a}\n"
            f"  Balance Entropy:  {bs_after.get('balance_entropy', 0):.4f}\n"
        )
    else:
        bal_text = (
            f"Class Balance Analysis\n"
            f"{'=' * 40}\n\n"
            f"(Not available — no\n"
            f" confidence filtering\n"
            f" was applied)\n"
        )
    ax_bal.set_title('Train Data Class Balance', fontweight='bold')
    ax_bal.text(0.05, 0.90, bal_text, transform=ax_bal.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#e3f2fd',
                          edgecolor='#90caf9', alpha=0.9))

    # ---------- PLOT 4: Entropy KDE ----------
    ax4 = fig.add_subplot(gs[1, 1])
    if weak_raw_available and entropy_weak_np.max() > 0:
        sns.kdeplot(entropy_weak_np, ax=ax4, color='#1f77b4', fill=True,
                    alpha=0.25, linewidth=2, label='Weak Model')
    sns.kdeplot(entropy_strong_np, ax=ax4, color='#ff7f0e', fill=True,
                alpha=0.25, linewidth=2, label='Strong Model (W2SG)')
    if weak_raw_available and entropy_weak_np.max() > 0:
        ax4.axvline(entropy_weak_np.mean(), color='#1f77b4', ls='--',
                    alpha=0.7, lw=1, label=f'Weak mean={entropy_weak_np.mean():.3f}')
    ax4.axvline(entropy_strong_np.mean(), color='#ff7f0e', ls='--',
                alpha=0.7, lw=1, label=f'Strong mean={entropy_strong_np.mean():.3f}')
    ax4.set_xlabel(r'Entropy  $H = -\sum p \log p$')
    ax4.set_ylabel('Density')
    ax4.set_title('Plot 4: Entropy Distribution (Weak vs Strong)', fontweight='bold')
    ax4.legend(loc='best', framealpha=0.85)
    ax4.set_xlim(left=0)

    # ---------- PLOT 5: Weak vs Strong Entropy Scatter ----------
    ax5 = fig.add_subplot(gs[1, 2])
    max_ent = max(entropy_weak_np.max(), entropy_strong_np.max()) * 1.15
    if max_ent == 0:
        max_ent = 1.0
    ax5.plot([0, max_ent], [0, max_ent], 'k--', lw=1, alpha=0.5, label='$y = x$')

    for mask, color, mrk, label, sz in group_specs:
        if mask.sum() > 0:
            ax5.scatter(entropy_weak_np[mask], entropy_strong_np[mask],
                        c=color, marker=mrk, s=sz, alpha=0.7,
                        label=f'{label} ({mask.sum()})',
                        edgecolors='black' if mrk in ('o', '^', '*') else 'none',
                        linewidths=0.3)
    ax5.set_xlabel('Weak Model Entropy')
    ax5.set_ylabel('Strong Model (W2SG) Entropy')
    ax5.set_title('Plot 5: Entropy Weak vs Strong (by Group)', fontweight='bold')
    ax5.legend(loc='best', framealpha=0.85, fontsize=7, markerscale=1.0)
    ax5.set_xlim(left=0); ax5.set_ylim(bottom=0)

    # ---------- PLOT 6 (slot): Summary Statistics ----------
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')

    n_total = len(y_true_np)
    summary = (
        f"W2SG Analysis Summary  (Seed {seed})\n"
        f"{'=' * 44}\n"
        f"Total test samples:  {n_total}\n"
        f"Dim. reduction:      {dim_label}\n\n"
        f"Group A (Both Correct):     {group_A.sum():5d}  "
        f"({group_A.sum() / n_total * 100:5.1f}%)\n"
        f"Group B (W2SG Phenomenon):  {group_B.sum():5d}  "
        f"({group_B.sum() / n_total * 100:5.1f}%)\n"
        f"Group C (Neg. Transfer):    {group_C.sum():5d}  "
        f"({group_C.sum() / n_total * 100:5.1f}%)\n"
        f"Group D (Both Wrong):       {group_D.sum():5d}  "
        f"({group_D.sum() / n_total * 100:5.1f}%)\n\n"
        f"Weak  Accuracy:  {weak_correct.sum() / n_total * 100:.1f}%\n"
        f"Strong Accuracy: {strong_correct.sum() / n_total * 100:.1f}%\n\n"
        f"Mean Entropy (Weak):    {entropy_weak_np.mean():.4f}\n"
        f"Mean Entropy (Strong):  {entropy_strong_np.mean():.4f}\n"
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffde7',
                       edgecolor='#bdbdbd', alpha=0.9))

    # ==================================================================
    # 9. Save
    # ==================================================================
    save_path = os.path.join(save_dir, f"seed{seed}_w2sg_{dim_label.lower()}.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    msg = f"[W2SG Plot] Saved to: {save_path}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return save_path



# =============================================================================
# Class balance statistics helper
# =============================================================================
def compute_class_balance_stats(labels, n_classes):
    """
    Compute class balance statistics for a label set.

    Args:
        labels: array-like of integer class labels (or soft labels to be argmax-ed).
        n_classes: total number of classes expected.

    Returns:
        dict with keys:
          - 'zero_sample_classes': number of classes with 0 samples
          - 'zero_class_ids': list of class ids with 0 samples
          - 'imbalance_ratio': max_count / min_nonzero_count (IR), inf if any class is 0
          - 'balance_entropy': normalized entropy of class distribution in [0, 1]
          - 'class_counts': dict mapping class_id -> count
          - 'n_total': total number of samples
    """
    if isinstance(labels, torch.Tensor):
        if labels.ndim > 1:
            labels = labels.argmax(-1)
        labels = labels.cpu().numpy()
    labels = np.asarray(labels).flatten()

    counts = np.zeros(n_classes, dtype=int)
    unique, cnts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, cnts):
        if 0 <= int(u) < n_classes:
            counts[int(u)] = c

    n_total = len(labels)
    zero_mask = counts == 0
    zero_sample_classes = int(zero_mask.sum())
    zero_class_ids = list(np.where(zero_mask)[0])

    nonzero_counts = counts[counts > 0]
    if len(nonzero_counts) == 0:
        ir = float('inf')
    elif len(nonzero_counts) == 1:
        ir = float('inf')  # only 1 class has samples
    else:
        ir = float(nonzero_counts.max()) / float(nonzero_counts.min())

    # Balance Degree Entropy: H(p) / log(K), where p_i = count_i / N
    if n_total > 0 and len(nonzero_counts) > 1:
        probs = counts / n_total
        probs_nz = probs[probs > 0]
        entropy_val = -np.sum(probs_nz * np.log(probs_nz))
        max_entropy = np.log(n_classes) if n_classes > 1 else 1.0
        balance_entropy = float(entropy_val / max_entropy)
    else:
        balance_entropy = 0.0

    class_counts = {int(i): int(counts[i]) for i in range(n_classes)}

    return {
        'zero_sample_classes': zero_sample_classes,
        'zero_class_ids': zero_class_ids,
        'imbalance_ratio': ir,
        'balance_entropy': balance_entropy,
        'class_counts': class_counts,
        'n_total': n_total,
    }


def format_balance_stats(stats, prefix=""):
    """Format balance stats as a multi-line string for logging."""
    lines = []
    lines.append(f"{prefix}Total samples: {stats['n_total']}")
    lines.append(f"{prefix}Zero-sample classes: {stats['zero_sample_classes']}"
                 f" (IDs: {stats['zero_class_ids'][:10]}{'...' if len(stats['zero_class_ids']) > 10 else ''})")
    ir_str = f"{stats['imbalance_ratio']:.2f}" if stats['imbalance_ratio'] != float('inf') else "inf"
    lines.append(f"{prefix}Imbalance Ratio (IR): {ir_str}")
    lines.append(f"{prefix}Balance Degree Entropy: {stats['balance_entropy']:.4f}")
    return "\n".join(lines)