import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from sklearn.mixture import GaussianMixture

# --- Các hàm helper GMM, mixup_data, sharpen giữ nguyên như cũ ---
def fit_gmm(losses, clean_threshold=0.8):
    losses = losses.reshape(-1, 1)
    loss_range = losses.max() - losses.min()
    if loss_range == 0: return np.ones(len(losses), dtype=bool), np.ones(len(losses))
    
    losses_norm = (losses - losses.min()) / loss_range
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(losses_norm)
    prob = gmm.predict_proba(losses_norm) 
    
    clean_idx = gmm.means_.argmin()
    clean_prob = prob[:, clean_idx]
    
    labeled_mask = clean_prob >= clean_threshold
    if labeled_mask.sum() < (0.01 * len(losses)): 
        labeled_mask = clean_prob >= 0.5 
    return labeled_mask, clean_prob

def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def sharpen(p, T=0.5):
    pt = p ** (1/T)
    return pt / pt.sum(dim=1, keepdim=True)

# [NEW] Hàm tính Entropy để phạt Over-confidence
def calc_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean()

# =========================================================================
def train_selfmix_probe(
    x_train,
    y_train,
    eval_datasets,
    device,
    loss_fn, # Vẫn nhận để không vỡ API
    n_epochs=20,
    weight_decay=0.0,
    lr=1e-3,
    batch_size=128,
    n_classes=1000,
    sample_weights=None,
    before_batch_callback=None,
    after_batch_callback=None,
    # --- Tham số SelfMix ---
    alpha=4.0,
    lambda_p=1.0,  # Trọng số cho Pseudo-Loss (Eq. 17)
    lambda_r=5.0,  # Trọng số cho R-Drop Loss (Eq. 17)
    lambda_e=0.8,  # [NEW] Trọng số phạt Confidence (Negative Entropy) lúc warmup
    dropout_rate=0.1,
    T=0.5,
    warmup_epochs=5
):
    ### setup training data
    x_train = x_train.float()
    
    if len(y_train.shape) == 2: y_train_hard = torch.argmax(y_train, dim=-1)
    elif len(y_train.shape) == 3: y_train_hard = y_train.mean(1).argmax(-1)
    else: y_train_hard = y_train
        
    train_ds = torch.utils.data.TensorDataset(
        x_train,
        y_train_hard,
        torch.arange(len(y_train_hard)),
        sample_weights if sample_weights is not None else torch.ones(len(y_train_hard), device=device),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    # ---------------------------------------------------------------------
    # THÊM DROPOUT VÀO MODEL ĐỂ HỖ TRỢ R-DROP (SELF-CONSISTENCY)
    # ---------------------------------------------------------------------
    model = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(x_train.shape[-1], n_classes)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    iter_i = 0
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)
    warning_printed = False

    ### train and eval
    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    results["clean_ratio"] = []
    
    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):
        model.eval() # Tắt dropout khi chạy GMM và Eval
        with torch.no_grad():
            all_logits = model(x_train.to(device))
            all_losses = F.cross_entropy(all_logits, y_train_hard.to(device), reduction='none').cpu().numpy()
            
            if epoch < warmup_epochs:
                labeled_mask = np.ones(len(x_train), dtype=bool)
                clean_ratio = 1.0
            else:
                labeled_mask, _ = fit_gmm(all_losses, clean_threshold=0.5)
                clean_ratio = labeled_mask.sum() / len(x_train)
                
            results["clean_ratio"].append(clean_ratio)
            
            all_pseudo_labels = sharpen(torch.softmax(all_logits, dim=-1), T=T)
            all_pseudo_hard = torch.argmax(all_logits, dim=-1)
            
        ### train
        model.train() # Bật lại dropout cho R-Drop
        correct, total = 0, 0
        for b_i, (x, y, sample_idxs, sample_ws) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(x)
            if before_batch_callback is not None:
                x, y, pred, sample_ws = before_batch_callback(x=x, y=y, pred=pred, sample_idxs=sample_idxs, sample_ws=sample_ws, epoch=epoch, is_eval=False)
            
            if pred.ndim > 2:
                pred = pred.mean(1)
            
            # =================================================================
            # PHÂN TÁCH LOGIC LOSS THEO EPOCH
            # =================================================================
            if epoch < warmup_epochs:
                # -------------------------------------------------------------
                # [NEW] WARM-UP PHASE: Standard CE + Confidence Penalty (Eq. 2)
                # -------------------------------------------------------------
                ce_loss = F.cross_entropy(pred, y)
                entropy = calc_entropy(pred)
                
                # Thêm Negative Entropy (-H) để phạt model quá tự tin
                loss = ce_loss - lambda_e * entropy
                
            else:
                # -------------------------------------------------------------
                # SELFMIX PHASE: GMM + Mix-Loss + Pseudo-Loss + R-Drop
                # -------------------------------------------------------------
                batch_mask = labeled_mask[sample_idxs.cpu().numpy()]
                mask_l = torch.tensor(batch_mask, device=device)
                mask_u = ~mask_l
                
                L_MIX = torch.tensor(0.0, device=device)
                L_P = torch.tensor(0.0, device=device)
                L_R = torch.tensor(0.0, device=device)
                
                # 1. MIX-LOSS
                loss_mix_l = torch.tensor(0.0, device=device)
                loss_mix_u = torch.tensor(0.0, device=device)
                
                if mask_l.sum() > 1:
                    x_l = x[mask_l]
                    y_l_onehot = F.one_hot(y[mask_l], num_classes=n_classes).float()
                    mixed_xl, yl_a, yl_b, lam_l = mixup_data(x_l, y_l_onehot, alpha, device)
                    
                    log_probs_l = F.log_softmax(model(mixed_xl), dim=1)
                    loss_mix_l = -lam_l * torch.mean(torch.sum(yl_a * log_probs_l, dim=1)) - (1 - lam_l) * torch.mean(torch.sum(yl_b * log_probs_l, dim=1))

                if mask_u.sum() > 1:
                    x_u = x[mask_u]
                    y_u_soft = all_pseudo_labels[sample_idxs][mask_u] 
                    
                    mixed_xu, yu_a, yu_b, lam_u = mixup_data(x_u, y_u_soft, alpha, device)
                    log_probs_u = F.log_softmax(model(mixed_xu), dim=1)
                    
                    loss_mix_u = -lam_u * torch.mean(torch.sum(yu_a * log_probs_u, dim=1)) - (1 - lam_u) * torch.mean(torch.sum(yu_b * log_probs_u, dim=1))
                
                L_MIX = loss_mix_l + loss_mix_u
                
                # 2. PSEUDO-LOSS & R-DROP
                if mask_u.sum() > 0:
                    x_u_raw = x[mask_u]
                    
                    # Pseudo-Loss
                    y_u_hard = all_pseudo_hard[sample_idxs][mask_u]
                    logits_u_raw = model(x_u_raw)
                    L_P = F.cross_entropy(logits_u_raw, y_u_hard)
                    
                    # R-Drop
                    logits_u_raw_2 = model(x_u_raw)
                    p1 = F.log_softmax(logits_u_raw, dim=-1)
                    p2 = F.softmax(logits_u_raw_2, dim=-1)
                    kl_1 = F.kl_div(p1, p2, reduction='batchmean')
                    
                    p2_log = F.log_softmax(logits_u_raw_2, dim=-1)
                    p1_prob = F.softmax(logits_u_raw, dim=-1)
                    kl_2 = F.kl_div(p2_log, p1_prob, reduction='batchmean')
                    
                    L_R = 0.5 * (kl_1 + kl_2)

                # TỔNG HỢP LOSS
                if L_MIX.item() == 0 and L_P.item() == 0:
                    loss = F.cross_entropy(pred, y)
                else:
                    rampup = min(1.0, (epoch - warmup_epochs) / max(1, (n_epochs - warmup_epochs)))
                    loss = L_MIX + (lambda_p * rampup * L_P) + (lambda_r * rampup * L_R)
            
            loss.backward()
            optimizer.step()
            schedule.step()
            iter_i += 1

            ### calc metrics
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
            if after_batch_callback is not None:
                after_batch_callback(x=x, y=y, pred=pred, loss=loss, last_in_epoch=b_i == n_batches - 1, epoch=epoch)
                
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f}, Clean {clean_ratio:.2f}")

        ### eval
        model.eval() 
        with torch.no_grad():
            for key, (x_test, y_test) in eval_datasets.items():
                x_test = x_test.float().to(device)
                pred = model(x_test).detach().cpu()
                if pred.ndim > 2: pred = pred.mean(1)
                if len(pred.shape) > 1: pred = torch.argmax(pred, dim=-1)
                
                y_test_eval = y_test
                if len(y_test_eval.shape) > 1: y_test_eval = torch.argmax(y_test_eval, dim=-1)
                if isinstance(y_test_eval, torch.Tensor): y_test_eval = y_test_eval.cpu()
                    
                acc = (pred == y_test_eval).float().mean()
                results[f"{key}_all"].append(acc)

    ### final results
    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]

    return results, model