import os
from copy import deepcopy
import numpy as np
import tqdm
import dill
import torch
from functools import partial
from datetime import datetime

from rw2s.utils import seed_all, preload
from rw2s.losses import LOSS_DICT

from rw2s.selfMix import fit_gmm, sharpen, train_selfmix_probe
from rw2s.plot import plot_w2sg_analysis, compute_class_balance_stats, format_balance_stats


def train_head(
    teacher_model,
    student_model,
    dataloader,
    cfg,
    logger,
    cached_labels_path,
    cached_embs_path,
    results,
    rng,
    n_classes,
    return_data=False,
    additional_eval_data=None,
    before_optim_run_callback_weak=None,
    before_optim_run_callback_gt=None,
    after_batch_callback_weak=None,
    before_batch_callback_weak=None,
    after_batch_callback_gt=None,
    before_batch_callback_gt=None,
):
    ### get (weak) labels from current teacher
    if cfg["w2s"]["load_labels"] and os.path.exists(cached_labels_path):
        # load from cache
        logger.info("Loading teacher labels from cache...")
        cached = torch.load(cached_labels_path, pickle_module=dill, map_location="cpu")
        gt_labels, teacher_labels, teacher_acc = cached["gt_labels"], cached["teacher_labels"], cached["teacher_acc"]
    else:
        # collect (and save)
        chunking_dir = os.path.join(os.path.dirname(cached_labels_path), f"label_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info("Collecting teacher labels...")
        teacher_embeddings, gt_labels, teacher_labels, teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=chunking_dir)
        if cfg["w2s"]["save_labels"]:
            torch.save({
                "cfg": cfg,
                "embeddings": teacher_embeddings,
                "inps": None,
                "gt_labels": gt_labels,
                "teacher_labels": teacher_labels,
                "teacher_acc": teacher_acc,
            }, cached_labels_path, pickle_module=dill)

    ### get embeddings from the student model
    if cfg["w2s"]["load_embeddings"] and os.path.exists(cached_embs_path):
        # load from cache
        logger.info("Loading student model embeddings from cache...")
        cached = torch.load(cached_embs_path, pickle_module=dill)
        student_embeddings, inps, student_gt_labels = cached["embeddings"], cached["inps"], cached["gt_labels"]
    else:
        # collect (and save)
        chunking_dir = os.path.join(os.path.dirname(cached_embs_path), f"embs_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info(f"Collecting embeddings (chunking directory: {chunking_dir})...")
        student_embeddings, student_gt_labels, student_labels, student_acc, inps, _ = preload(model=student_model, loader=dataloader, device=cfg["device"], chunking_dir=chunking_dir, store_embs=True)
        if cfg["w2s"]["save_embeddings"]:
            torch.save({
                "cfg": cfg,
                "embeddings": student_embeddings,
                "inps": inps,
                "gt_labels": student_gt_labels,
                "student_labels": student_labels,
                "student_acc": student_acc,
            }, cached_embs_path, pickle_module=dill)
    assert torch.all(gt_labels == student_gt_labels), "GT labels from teacher and student do not match."
    del student_gt_labels

    ### order of samples
    order = np.arange(len(gt_labels))
    rng.shuffle(order)
    results["order"].append(order)

    ### all data
    x = student_embeddings[order]
    y = gt_labels[order]
    yw = teacher_labels[order]

    ### split
    assert len(cfg["w2s"]["train_val_test_split"]) == 3, "Train, val, test split must be of length 3."
    assert sum(cfg["w2s"]["train_val_test_split"]) == 1.0, "Train, val, test split must sum to 1."
    n_train, n_val = int(cfg["w2s"]["train_val_test_split"][0] * len(x)), int(cfg["w2s"]["train_val_test_split"][1] * len(x))
    x_train, x_val, x_test = x[:n_train], x[n_train:n_train+n_val], x[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    yw_train, yw_val, yw_test = yw[:n_train], yw[n_train:n_train+n_val], yw[n_train+n_val:]
    yw_val_raw = yw_val.mean(1) if yw_val.ndim == 3 else yw_val
    yw_val = yw_val_raw.argmax(-1) # only for evaluation
    yw_test_raw = yw_test.mean(1) if yw_test.ndim == 3 else yw_test
    yw_test = yw_test_raw.argmax(-1) # only for evaluation
    
    eval_datasets = {"val": (x_val, y_val), "val_weak": (x_val, yw_val), "val_weak_raw": (x_val, yw_val_raw), "test": (x_test, y_test), "test_weak": (x_test, yw_test), "test_weak_raw": (x_test, yw_test_raw)}
    if additional_eval_data is not None:
        for k, v in additional_eval_data.items():
            eval_datasets[k] = v
    logger.info(f"\nTotal number of samples: {len(x)}.")
    logger.info(f"  Number of training samples: {len(x_train)}.")
    logger.info(f"  Number of validation samples: {len(x_val)}.")
    logger.info(f"  Number of testing samples: {len(x_test)}.")

    ### eval teacher (average weak labels)
    results["teacher_acc_src"].append(teacher_acc)
    teacher_acc_all = (y == (yw if yw.ndim == 2 else yw.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc"].append(teacher_acc_all)
    teacher_acc_train = (y_train == (yw_train if yw_train.ndim == 2 else yw_train.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc_train"].append(teacher_acc_train)
    teacher_acc_val = (y_val == yw_val).float().mean()
    results["teacher_acc_val"].append(teacher_acc_val)
    teacher_acc_test = (y_test == yw_test).float().mean()
    results["teacher_acc_test"].append(teacher_acc_test)
    if type(teacher_acc) == float:
        teacher_acc = torch.tensor([teacher_acc], device=cfg["device"])
    logger.info(f"Teacher label accuracy (all data, not combined): {[np.round(tacc.item() if hasattr(tacc, 'item') else tacc, 4) for tacc in teacher_acc]}")
    logger.info(f"Teacher label accuracy (all data): {teacher_acc_all:.4f}")
    logger.info(f"Teacher label accuracy (train): {teacher_acc_train:.4f}")
    logger.info(f"Teacher label accuracy (val): {teacher_acc_val:.4f}")
    logger.info(f"Teacher label accuracy (test): {teacher_acc_test:.4f}")

    ### w2s
    if before_optim_run_callback_weak is not None:
        before_optim_run_callback_weak(yw=yw_train, sample_idxs=np.arange(len(yw_train)))
    seed_all(cfg["seed"]) # important to get same results for cached/not cached
    results_teacher_to_student, student_model_probe = train_logreg(x_train, yw_train, eval_datasets, device=cfg["device"],
        batch_size=cfg["w2s"]["batch_size"], loss_fn=LOSS_DICT[cfg["w2s"]["teacher_labels_loss_fn_name"]](**(cfg["w2s"]["teacher_labels_loss_fn_kwargs"] or dict())), n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None, before_batch_callback=before_batch_callback_weak, after_batch_callback=after_batch_callback_weak)
    results["results_teacher_to_student"].append(results_teacher_to_student)
    results["student_model_probe"].append(student_model_probe)

    ### gt
    if before_optim_run_callback_gt is not None:
        before_optim_run_callback_gt(yw=yw_test, sample_idxs=n_train + np.arange(len(yw_test)))
    seed_all(cfg["seed"])
    results_gt, _ = train_logreg(x_train, y_train, eval_datasets, device=cfg["device"], batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["gt_labels_loss_fn_name"]](**(cfg["w2s"]["gt_labels_loss_fn_kwargs"] or dict())), n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None, before_batch_callback=before_batch_callback_gt, after_batch_callback=after_batch_callback_gt)
    results["results_gt"].append(results_gt)

    if return_data:
        return results, student_model_probe, {"x": x, "y": y, "yw": yw, "x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val, "x_test": x_test, "y_test": y_test, "yw_train": yw_train, "yw_val": yw_val, "yw_test": yw_test}
    return results, student_model_probe


import os
import torch
import numpy as np
import dill
from datetime import datetime
from functools import partial

def train_head_DG(
    teacher_model,
    student_model,
    val_dataloader,     # CHANGED: Thay dataloader bằng val_dataloader
    test_dataloader,    # NEW: Thêm test_dataloader
    cfg,
    logger,
    cached_labels_path,
    cached_embs_path,
    results,
    rng,
    n_classes,
    return_data=False,
    additional_eval_data=None,
    before_optim_run_callback_weak=None,
    before_optim_run_callback_gt=None,
    after_batch_callback_weak=None,
    before_batch_callback_weak=None,
    after_batch_callback_gt=None,
    before_batch_callback_gt=None,
):
    """
    Thiết kế đầu vào nhận 2 domain val và test
    chia val thành 0.8, 0.2: train, val cho w2s
    test: target data lấy toàn bộ
    """
    
    ### get (weak) labels from current teacher
    if cfg["w2s"]["load_labels"] and os.path.exists(cached_labels_path):
        # load from cache
        logger.info("Loading teacher labels from cache...")
        cached = torch.load(cached_labels_path, pickle_module=dill, map_location="cpu")
        val_gt_labels, val_teacher_labels = cached["val_gt_labels"], cached["val_teacher_labels"]
        test_gt_labels, test_teacher_labels = cached["test_gt_labels"], cached["test_teacher_labels"]
        teacher_acc = cached.get("teacher_acc", 0) # Fallback if not saved
    else:
        # collect (and save) for BOTH val and test
        chunking_dir = os.path.join(os.path.dirname(cached_labels_path), f"label_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info("Collecting teacher labels for Validation Set...")
        _, val_gt_labels, val_teacher_labels, val_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=val_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'val'))
        
        logger.info("Collecting teacher labels for Test Set...")
        _, test_gt_labels, test_teacher_labels, test_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=test_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'test'))
        
        # Trích xuất giá trị số thực (float) an toàn trước khi tính trung bình
        val_acc_float = np.mean(val_teacher_acc)
        test_acc_float = np.mean(test_teacher_acc)
        teacher_acc = float((val_acc_float + test_acc_float) / 2.0)

        if cfg["w2s"]["save_labels"]:
            torch.save({
                "cfg": cfg,
                "val_gt_labels": val_gt_labels,
                "val_teacher_labels": val_teacher_labels,
                "test_gt_labels": test_gt_labels,
                "test_teacher_labels": test_teacher_labels,
                "teacher_acc": teacher_acc,
            }, cached_labels_path, pickle_module=dill)

    ### get embeddings from the student model
    if cfg["w2s"]["load_embeddings"] and os.path.exists(cached_embs_path):
        # load from cache
        logger.info("Loading student model embeddings from cache...")
        cached = torch.load(cached_embs_path, pickle_module=dill)
        val_student_embeddings, val_student_gt_labels = cached["val_embeddings"], cached["val_gt_labels"]
        test_student_embeddings, test_student_gt_labels = cached["test_embeddings"], cached["test_gt_labels"]
    else:
        # collect (and save) for BOTH val and test
        chunking_dir = os.path.join(os.path.dirname(cached_embs_path), f"embs_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        
        logger.info(f"Collecting student embeddings for Validation Set...")
        val_student_embeddings, val_student_gt_labels, _, _, _, _ = preload(model=student_model, loader=val_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'val'), store_embs=True)
        
        logger.info(f"Collecting student embeddings for Test Set...")
        test_student_embeddings, test_student_gt_labels, _, _, _, _ = preload(model=student_model, loader=test_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'test'), store_embs=True)

        if cfg["w2s"]["save_embeddings"]:
            torch.save({
                "cfg": cfg,
                "val_embeddings": val_student_embeddings,
                "val_gt_labels": val_student_gt_labels,
                "test_embeddings": test_student_embeddings,
                "test_gt_labels": test_student_gt_labels,
            }, cached_embs_path, pickle_module=dill)
    # breakpoint()
    assert torch.all(val_gt_labels == val_student_gt_labels), "Val GT labels from teacher and student do not match."
    # breakpoint()
    assert torch.all(test_gt_labels == test_student_gt_labels), "Test GT labels from teacher and student do not match."
    del val_student_gt_labels, test_student_gt_labels

    ### CHANGED: Trộn (shuffle) và cắt dữ liệu chỉ cho tập VAL
    order = np.arange(len(val_gt_labels))
    rng.shuffle(order)
    results["order"].append(order)

    x_val_all = val_student_embeddings[order]
    y_val_all = val_gt_labels[order]
    yw_val_all = val_teacher_labels[order]

    ### split Validation into Train & Val for w2s
    assert len(cfg["w2s"]["train_val_test_split_DG"]) == 2, "Train/val split config must be of length 2 (e.g., [0.8, 0.2])."
    assert sum(cfg["w2s"]["train_val_test_split_DG"]) == 1.0, "Train/val split must sum to 1."
    
    n_train = int(cfg["w2s"]["train_val_test_split_DG"][0] * len(x_val_all))
    
    # Lấy Train & Val từ val_dataloader
    x_train, x_val = x_val_all[:n_train], x_val_all[n_train:]
    y_train, y_val = y_val_all[:n_train], y_val_all[n_train:]
    yw_train, yw_val = yw_val_all[:n_train], yw_val_all[n_train:]

    # Lấy Test toàn bộ từ test_dataloader
    x_test = test_student_embeddings
    y_test = test_gt_labels
    yw_test = test_teacher_labels

    # Nối lại để tính toán logging chung
    x = torch.cat([x_train, x_val, x_test]) if isinstance(x_train, torch.Tensor) else np.concatenate([x_train, x_val, x_test])
    y = torch.cat([y_train, y_val, y_test]) if isinstance(y_train, torch.Tensor) else np.concatenate([y_train, y_val, y_test])
    yw = torch.cat([yw_train, yw_val, yw_test]) if isinstance(yw_train, torch.Tensor) else np.concatenate([yw_train, yw_val, yw_test])

    yw_val_raw = yw_val.mean(1) if yw_val.ndim == 3 else yw_val
    yw_val = yw_val_raw.argmax(-1) # only for evaluation
    yw_test_raw = yw_test.mean(1) if yw_test.ndim == 3 else yw_test
    yw_test = yw_test_raw.argmax(-1) # only for evaluation
    
    eval_datasets = {"val": (x_val, y_val), "val_weak": (x_val, yw_val), "val_weak_raw": (x_val, yw_val_raw), "test": (x_test, y_test), "test_weak": (x_test, yw_test), "test_weak_raw": (x_test, yw_test_raw)}
    if additional_eval_data is not None:
        for k, v in additional_eval_data.items():
            eval_datasets[k] = v
            
    logger.info(f"\nTotal number of samples: {len(x)}.")
    logger.info(f"  Number of training samples (from Val Dataloader): {len(x_train)}.")
    logger.info(f"  Number of validation samples (from Val Dataloader): {len(x_val)}.")
    logger.info(f"  Number of testing samples (from Test Dataloader): {len(x_test)}.")

    ### eval teacher (average weak labels)
    results["teacher_acc_src"].append(teacher_acc)
    teacher_acc_all = (y == (yw if yw.ndim == 2 else yw.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc"].append(teacher_acc_all)
    teacher_acc_train = (y_train == (yw_train if yw_train.ndim == 2 else yw_train.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc_train"].append(teacher_acc_train)
    teacher_acc_val = (y_val == yw_val).float().mean()
    results["teacher_acc_val"].append(teacher_acc_val)
    teacher_acc_test = (y_test == yw_test).float().mean()
    results["teacher_acc_test"].append(teacher_acc_test)
    
    if type(teacher_acc) == float:
        teacher_acc = torch.tensor([teacher_acc], device=cfg["device"])
    logger.info(f"Teacher label accuracy (all data, not combined): {[np.round(tacc.item() if hasattr(tacc, 'item') else tacc, 4) for tacc in teacher_acc]}")
    logger.info(f"Teacher label accuracy (all data): {teacher_acc_all:.4f}")
    logger.info(f"Teacher label accuracy (train): {teacher_acc_train:.4f}")
    logger.info(f"Teacher label accuracy (val): {teacher_acc_val:.4f}")
    logger.info(f"Teacher label accuracy (test): {teacher_acc_test:.4f}")

    ### w2s
    if before_optim_run_callback_weak is not None:
        before_optim_run_callback_weak(yw=yw_train, sample_idxs=np.arange(len(yw_train)))
    seed_all(cfg["seed"]) # important to get same results for cached/not cached
    results_teacher_to_student, student_model_probe = train_logreg(x_train, yw_train, eval_datasets, device=cfg["device"],
        batch_size=cfg["w2s"]["batch_size"], loss_fn=LOSS_DICT[cfg["w2s"]["teacher_labels_loss_fn_name"]](**(cfg["w2s"]["teacher_labels_loss_fn_kwargs"] or dict())), n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None, before_batch_callback=before_batch_callback_weak, after_batch_callback=after_batch_callback_weak)
    results["results_teacher_to_student"].append(results_teacher_to_student)
    results["student_model_probe"].append(student_model_probe)

    ### gt
    if before_optim_run_callback_gt is not None:
        # CHANGED: Index của tập test bị dời đi một đoạn bằng tổng số mẫu của train và val
        before_optim_run_callback_gt(yw=yw_test, sample_idxs=len(y_train) + len(y_val) + np.arange(len(yw_test)))
    seed_all(cfg["seed"])
    results_gt, gt_model_probe = train_logreg(x_train, y_train, eval_datasets, device=cfg["device"], batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["gt_labels_loss_fn_name"]](**(cfg["w2s"]["gt_labels_loss_fn_kwargs"] or dict())), n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None, before_batch_callback=before_batch_callback_gt, after_batch_callback=after_batch_callback_gt)
    results["results_gt"].append(results_gt)

    ### W2SG visualization (toggleable) — after both w2sg and gt training
    if cfg["w2s"].get("plot_w2sg", False):
        _plot_dir = cfg["w2s"].get("plot_save_dir", None)
        if _plot_dir:
            _dim_methods = cfg["w2s"].get("plot_dim_reduction", ["pca"])
            if isinstance(_dim_methods, str):
                _dim_methods = [_dim_methods]
            for _dm in _dim_methods:
                try:
                    plot_w2sg_analysis(
                        model=student_model_probe,
                        gt_model=gt_model_probe,
                        eval_datasets=eval_datasets,
                        save_dir=_plot_dir,
                        seed=cfg["seed"],
                        device=cfg["device"],
                        dim_reduction=_dm,
                        logger=logger,
                    )
                except Exception as _plot_err:
                    logger.info(f"[W2SG Plot] Warning: plotting failed ({_dm}): {_plot_err}")

    if return_data:
        return results, student_model_probe, {"x": x, "y": y, "yw": yw, "x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val, "x_test": x_test, "y_test": y_test, "yw_train": yw_train, "yw_val": yw_val, "yw_test": yw_test}
    return results, student_model_probe



def train_head_DG_hard(
    teacher_model,
    student_model,
    val_dataloader,
    test_dataloader,
    cfg,
    logger,
    cached_labels_path,
    cached_embs_path,
    results,
    rng,
    n_classes,
    return_data=False,
    additional_eval_data=None,
    before_optim_run_callback_weak=None,
    before_optim_run_callback_gt=None,
    after_batch_callback_weak=None,
    before_batch_callback_weak=None,
    after_batch_callback_gt=None,
    before_batch_callback_gt=None,
):
    """
    train_head_DG with hard sample curriculum learning.
    Same as train_head_DG but uses train_logreg_hard for w2s training:
      1. Filters training data by weak model confidence clustering
      2. Warmup epochs with equal weights
      3. Then splits into easy/hard by w2sg confidence, upweights hard samples

    Extra cfg keys (all under cfg["w2s"]):
      - warmup_hard_epochs (int, default 3): number of warmup epochs
      - hard_weight (float, default 2.0): weight multiplier for hard samples
      - cluster_method (str, default 'gmm'): 'gmm', 'kmeans', 'threshold', 'median'
      - weak_conf_threshold (float, default 0.5): threshold for weak confidence clustering
      - w2sg_conf_threshold (float, default 0.5): threshold for w2sg confidence clustering
    """

    ### get (weak) labels from current teacher
    if cfg["w2s"]["load_labels"] and os.path.exists(cached_labels_path):
        # load from cache
        logger.info("Loading teacher labels from cache...")
        cached = torch.load(cached_labels_path, pickle_module=dill, map_location="cpu")
        val_gt_labels, val_teacher_labels = cached["val_gt_labels"], cached["val_teacher_labels"]
        test_gt_labels, test_teacher_labels = cached["test_gt_labels"], cached["test_teacher_labels"]
        teacher_acc = cached.get("teacher_acc", 0)
    else:
        # collect (and save) for BOTH val and test
        chunking_dir = os.path.join(os.path.dirname(cached_labels_path), f"label_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info("Collecting teacher labels for Validation Set...")
        _, val_gt_labels, val_teacher_labels, val_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=val_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'val'))

        logger.info("Collecting teacher labels for Test Set...")
        _, test_gt_labels, test_teacher_labels, test_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=test_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'test'))

        val_acc_float = np.mean(val_teacher_acc)
        test_acc_float = np.mean(test_teacher_acc)
        teacher_acc = float((val_acc_float + test_acc_float) / 2.0)

        if cfg["w2s"]["save_labels"]:
            torch.save({
                "cfg": cfg,
                "val_gt_labels": val_gt_labels,
                "val_teacher_labels": val_teacher_labels,
                "test_gt_labels": test_gt_labels,
                "test_teacher_labels": test_teacher_labels,
                "teacher_acc": teacher_acc,
            }, cached_labels_path, pickle_module=dill)

    ### get embeddings from the student model
    if cfg["w2s"]["load_embeddings"] and os.path.exists(cached_embs_path):
        # load from cache
        logger.info("Loading student model embeddings from cache...")
        cached = torch.load(cached_embs_path, pickle_module=dill)
        val_student_embeddings, val_student_gt_labels = cached["val_embeddings"], cached["val_gt_labels"]
        test_student_embeddings, test_student_gt_labels = cached["test_embeddings"], cached["test_gt_labels"]
    else:
        # collect (and save) for BOTH val and test
        chunking_dir = os.path.join(os.path.dirname(cached_embs_path), f"embs_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)

        logger.info(f"Collecting student embeddings for Validation Set...")
        val_student_embeddings, val_student_gt_labels, _, _, _, _ = preload(model=student_model, loader=val_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'val'), store_embs=True)

        logger.info(f"Collecting student embeddings for Test Set...")
        test_student_embeddings, test_student_gt_labels, _, _, _, _ = preload(model=student_model, loader=test_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'test'), store_embs=True)

        if cfg["w2s"]["save_embeddings"]:
            torch.save({
                "cfg": cfg,
                "val_embeddings": val_student_embeddings,
                "val_gt_labels": val_student_gt_labels,
                "test_embeddings": test_student_embeddings,
                "test_gt_labels": test_student_gt_labels,
            }, cached_embs_path, pickle_module=dill)

    assert torch.all(val_gt_labels == val_student_gt_labels), "Val GT labels from teacher and student do not match."
    assert torch.all(test_gt_labels == test_student_gt_labels), "Test GT labels from teacher and student do not match."
    del val_student_gt_labels, test_student_gt_labels

    ### Shuffle and split VAL data
    order = np.arange(len(val_gt_labels))
    rng.shuffle(order)
    results["order"].append(order)

    x_val_all = val_student_embeddings[order]
    y_val_all = val_gt_labels[order]
    yw_val_all = val_teacher_labels[order]

    ### split Validation into Train & Val for w2s
    assert len(cfg["w2s"]["train_val_test_split_DG"]) == 2, "Train/val split config must be of length 2 (e.g., [0.8, 0.2])."
    assert sum(cfg["w2s"]["train_val_test_split_DG"]) == 1.0, "Train/val split must sum to 1."

    n_train = int(cfg["w2s"]["train_val_test_split_DG"][0] * len(x_val_all))

    x_train, x_val = x_val_all[:n_train], x_val_all[n_train:]
    y_train, y_val = y_val_all[:n_train], y_val_all[n_train:]
    yw_train, yw_val = yw_val_all[:n_train], yw_val_all[n_train:]

    # Test set from test_dataloader
    x_test = test_student_embeddings
    y_test = test_gt_labels
    yw_test = test_teacher_labels

    # Concat for logging
    x = torch.cat([x_train, x_val, x_test]) if isinstance(x_train, torch.Tensor) else np.concatenate([x_train, x_val, x_test])
    y = torch.cat([y_train, y_val, y_test]) if isinstance(y_train, torch.Tensor) else np.concatenate([y_train, y_val, y_test])
    yw = torch.cat([yw_train, yw_val, yw_test]) if isinstance(yw_train, torch.Tensor) else np.concatenate([yw_train, yw_val, yw_test])

    yw_val_raw = yw_val.mean(1) if yw_val.ndim == 3 else yw_val
    yw_val = yw_val_raw.argmax(-1)
    yw_test_raw = yw_test.mean(1) if yw_test.ndim == 3 else yw_test
    yw_test = yw_test_raw.argmax(-1)
    
    eval_datasets = {"val": (x_val, y_val), "val_weak": (x_val, yw_val), "val_weak_raw": (x_val, yw_val_raw), "test": (x_test, y_test), "test_weak": (x_test, yw_test), "test_weak_raw": (x_test, yw_test_raw)}
    if additional_eval_data is not None:
        for k, v in additional_eval_data.items():
            eval_datasets[k] = v

    logger.info(f"\nTotal number of samples: {len(x)}.")
    logger.info(f"  Number of training samples (from Val Dataloader): {len(x_train)}.")
    logger.info(f"  Number of validation samples (from Val Dataloader): {len(x_val)}.")
    logger.info(f"  Number of testing samples (from Test Dataloader): {len(x_test)}.")

    ### eval teacher (average weak labels)
    results["teacher_acc_src"].append(teacher_acc)
    teacher_acc_all = (y == (yw if yw.ndim == 2 else yw.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc"].append(teacher_acc_all)
    teacher_acc_train = (y_train == (yw_train if yw_train.ndim == 2 else yw_train.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc_train"].append(teacher_acc_train)
    teacher_acc_val = (y_val == yw_val).float().mean()
    results["teacher_acc_val"].append(teacher_acc_val)
    teacher_acc_test = (y_test == yw_test).float().mean()
    results["teacher_acc_test"].append(teacher_acc_test)

    if type(teacher_acc) == float:
        teacher_acc = torch.tensor([teacher_acc], device=cfg["device"])
    logger.info(f"Teacher label accuracy (all data, not combined): {[np.round(tacc.item() if hasattr(tacc, 'item') else tacc, 4) for tacc in teacher_acc]}")
    logger.info(f"Teacher label accuracy (all data): {teacher_acc_all:.4f}")
    logger.info(f"Teacher label accuracy (train): {teacher_acc_train:.4f}")
    logger.info(f"Teacher label accuracy (val): {teacher_acc_val:.4f}")
    logger.info(f"Teacher label accuracy (test): {teacher_acc_test:.4f}")

    ### w2s with hard sample curriculum
    if before_optim_run_callback_weak is not None:
        before_optim_run_callback_weak(yw=yw_train, sample_idxs=np.arange(len(yw_train)))
    seed_all(cfg["seed"])

    # Read hard curriculum params from cfg (with defaults)
    hard_curriculum_kwargs = {
        'warmup_hard_epochs': cfg["w2s"].get("warmup_hard_epochs", 45),
        'hard_weight': cfg["w2s"].get("hard_weight", 2.0),
        'cluster_method': cfg["w2s"].get("cluster_method", "kmeans"),
        'weak_conf_threshold': cfg["w2s"].get("weak_conf_threshold", 0.6),
        'w2sg_conf_threshold': cfg["w2s"].get("w2sg_conf_threshold", 0.6),
    }
    logger.info(f"Hard curriculum params: {hard_curriculum_kwargs}")

    results_teacher_to_student, student_model_probe = train_logreg_hard(
        x_train, yw_train, eval_datasets, device=cfg["device"],
        batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["teacher_labels_loss_fn_name"]](**(cfg["w2s"]["teacher_labels_loss_fn_kwargs"] or dict())),
        n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None,
        before_batch_callback=before_batch_callback_weak,
        after_batch_callback=after_batch_callback_weak,
        **hard_curriculum_kwargs,
    )
    results["results_teacher_to_student"].append(results_teacher_to_student)
    results["student_model_probe"].append(student_model_probe)

    ### gt
    if before_optim_run_callback_gt is not None:
        before_optim_run_callback_gt(yw=yw_test, sample_idxs=len(y_train) + len(y_val) + np.arange(len(yw_test)))
    seed_all(cfg["seed"])
    results_gt, gt_model_probe = train_logreg(x_train, y_train, eval_datasets, device=cfg["device"], batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["gt_labels_loss_fn_name"]](**(cfg["w2s"]["gt_labels_loss_fn_kwargs"] or dict())), n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None, before_batch_callback=before_batch_callback_gt, after_batch_callback=after_batch_callback_gt)
    results["results_gt"].append(results_gt)

    ### W2SG visualization (toggleable) — after both w2sg and gt training
    _balance_stats = None
    if '_balance_before' in results_teacher_to_student and '_balance_after' in results_teacher_to_student:
        _balance_stats = {
            'before': results_teacher_to_student['_balance_before'],
            'after': results_teacher_to_student['_balance_after'],
        }
    if cfg["w2s"].get("plot_w2sg", False):
        _plot_dir = cfg["w2s"].get("plot_save_dir", None)
        if _plot_dir:
            _dim_methods = cfg["w2s"].get("plot_dim_reduction", ["pca"])
            if isinstance(_dim_methods, str):
                _dim_methods = [_dim_methods]
            for _dm in _dim_methods:
                try:
                    plot_w2sg_analysis(
                        model=student_model_probe,
                        gt_model=gt_model_probe,
                        eval_datasets=eval_datasets,
                        save_dir=_plot_dir,
                        seed=cfg["seed"],
                        device=cfg["device"],
                        dim_reduction=_dm,
                        balance_stats=_balance_stats,
                        logger=logger,
                    )
                except Exception as _plot_err:
                    logger.info(f"[W2SG Plot] Warning: plotting failed ({_dm}): {_plot_err}")

    if return_data:
        return results, student_model_probe, {"x": x, "y": y, "yw": yw, "x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val, "x_test": x_test, "y_test": y_test, "yw_train": yw_train, "yw_val": yw_val, "yw_test": yw_test}
    return results, student_model_probe



def train_head_entropy_hard(
    teacher_model,
    student_model,
    val_dataloader,
    test_dataloader,
    cfg,
    logger,
    cached_labels_path,
    cached_embs_path,
    results,
    rng,
    n_classes,
    return_data=False,
    additional_eval_data=None,
    before_optim_run_callback_weak=None,
    before_optim_run_callback_gt=None,
    after_batch_callback_weak=None,
    before_batch_callback_weak=None,
    after_batch_callback_gt=None,
    before_batch_callback_gt=None,
):
    """
    train_head_DG with hard sample + entropy curriculum learning.
    Same as train_head_DG_hard but uses train_logreg_entropy_hard for w2s training:
      1. Filters training data by weak model confidence clustering
      2. Warmup epochs with equal weights
      3. Then splits into D0 (easy) / D1 (hard) by w2sg confidence,
         adds D2 (high entropy from original data), and upweights D1 > D0, D2 > D1

    Extra cfg keys (all under cfg["w2s"]):
      - warmup_hard_epochs (int, default 5): number of warmup epochs
      - cluster_method (str, default 'gmm'): 'gmm', 'kmeans', 'threshold', 'median'
      - weak_conf_threshold (float, default 0.6): threshold for weak confidence clustering
      - w2sg_conf_threshold (float, default 0.6): threshold for w2sg confidence clustering
      - entropy_threshold (float, default 1.5): w2sg entropy threshold for D2
      - d0_weight (float, default 1.0): weight for D0 (easy)
      - d1_weight (float, default 2.0): weight for D1 (hard)
      - d2_weight (float, default 3.0): weight for D2 (high entropy)
    """

    ### get (weak) labels from current teacher
    if cfg["w2s"]["load_labels"] and os.path.exists(cached_labels_path):
        logger.info("Loading teacher labels from cache...")
        cached = torch.load(cached_labels_path, pickle_module=dill, map_location="cpu")
        val_gt_labels, val_teacher_labels = cached["val_gt_labels"], cached["val_teacher_labels"]
        test_gt_labels, test_teacher_labels = cached["test_gt_labels"], cached["test_teacher_labels"]
        teacher_acc = cached.get("teacher_acc", 0)
    else:
        chunking_dir = os.path.join(os.path.dirname(cached_labels_path), f"label_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info("Collecting teacher labels for Validation Set...")
        _, val_gt_labels, val_teacher_labels, val_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=val_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'val'))

        logger.info("Collecting teacher labels for Test Set...")
        _, test_gt_labels, test_teacher_labels, test_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=test_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'test'))

        val_acc_float = np.mean(val_teacher_acc)
        test_acc_float = np.mean(test_teacher_acc)
        teacher_acc = float((val_acc_float + test_acc_float) / 2.0)

        if cfg["w2s"]["save_labels"]:
            torch.save({
                "cfg": cfg,
                "val_gt_labels": val_gt_labels,
                "val_teacher_labels": val_teacher_labels,
                "test_gt_labels": test_gt_labels,
                "test_teacher_labels": test_teacher_labels,
                "teacher_acc": teacher_acc,
            }, cached_labels_path, pickle_module=dill)

    ### get embeddings from the student model
    if cfg["w2s"]["load_embeddings"] and os.path.exists(cached_embs_path):
        logger.info("Loading student model embeddings from cache...")
        cached = torch.load(cached_embs_path, pickle_module=dill)
        val_student_embeddings, val_student_gt_labels = cached["val_embeddings"], cached["val_gt_labels"]
        test_student_embeddings, test_student_gt_labels = cached["test_embeddings"], cached["test_gt_labels"]
    else:
        chunking_dir = os.path.join(os.path.dirname(cached_embs_path), f"embs_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)

        logger.info(f"Collecting student embeddings for Validation Set...")
        val_student_embeddings, val_student_gt_labels, _, _, _, _ = preload(model=student_model, loader=val_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'val'), store_embs=True)

        logger.info(f"Collecting student embeddings for Test Set...")
        test_student_embeddings, test_student_gt_labels, _, _, _, _ = preload(model=student_model, loader=test_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'test'), store_embs=True)

        if cfg["w2s"]["save_embeddings"]:
            torch.save({
                "cfg": cfg,
                "val_embeddings": val_student_embeddings,
                "val_gt_labels": val_student_gt_labels,
                "test_embeddings": test_student_embeddings,
                "test_gt_labels": test_student_gt_labels,
            }, cached_embs_path, pickle_module=dill)

    assert torch.all(val_gt_labels == val_student_gt_labels), "Val GT labels from teacher and student do not match."
    assert torch.all(test_gt_labels == test_student_gt_labels), "Test GT labels from teacher and student do not match."
    del val_student_gt_labels, test_student_gt_labels

    ### Shuffle and split VAL data
    order = np.arange(len(val_gt_labels))
    rng.shuffle(order)
    results["order"].append(order)

    x_val_all = val_student_embeddings[order]
    y_val_all = val_gt_labels[order]
    yw_val_all = val_teacher_labels[order]

    ### split Validation into Train & Val for w2s
    assert len(cfg["w2s"]["train_val_test_split_DG"]) == 2, "Train/val split config must be of length 2 (e.g., [0.8, 0.2])."
    assert sum(cfg["w2s"]["train_val_test_split_DG"]) == 1.0, "Train/val split must sum to 1."

    n_train = int(cfg["w2s"]["train_val_test_split_DG"][0] * len(x_val_all))

    x_train, x_val = x_val_all[:n_train], x_val_all[n_train:]
    y_train, y_val = y_val_all[:n_train], y_val_all[n_train:]
    yw_train, yw_val = yw_val_all[:n_train], yw_val_all[n_train:]

    # Test set from test_dataloader
    x_test = test_student_embeddings
    y_test = test_gt_labels
    yw_test = test_teacher_labels

    # Concat for logging
    x = torch.cat([x_train, x_val, x_test]) if isinstance(x_train, torch.Tensor) else np.concatenate([x_train, x_val, x_test])
    y = torch.cat([y_train, y_val, y_test]) if isinstance(y_train, torch.Tensor) else np.concatenate([y_train, y_val, y_test])
    yw = torch.cat([yw_train, yw_val, yw_test]) if isinstance(yw_train, torch.Tensor) else np.concatenate([yw_train, yw_val, yw_test])

    yw_val_raw = yw_val.mean(1) if yw_val.ndim == 3 else yw_val
    yw_val = yw_val_raw.argmax(-1)
    yw_test_raw = yw_test.mean(1) if yw_test.ndim == 3 else yw_test
    yw_test = yw_test_raw.argmax(-1)

    eval_datasets = {"val": (x_val, y_val), "val_weak": (x_val, yw_val), "val_weak_raw": (x_val, yw_val_raw), "test": (x_test, y_test), "test_weak": (x_test, yw_test), "test_weak_raw": (x_test, yw_test_raw)}
    if additional_eval_data is not None:
        for k, v in additional_eval_data.items():
            eval_datasets[k] = v

    logger.info(f"\nTotal number of samples: {len(x)}.")
    logger.info(f"  Number of training samples (from Val Dataloader): {len(x_train)}.")
    logger.info(f"  Number of validation samples (from Val Dataloader): {len(x_val)}.")
    logger.info(f"  Number of testing samples (from Test Dataloader): {len(x_test)}.")

    ### eval teacher (average weak labels)
    results["teacher_acc_src"].append(teacher_acc)
    teacher_acc_all = (y == (yw if yw.ndim == 2 else yw.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc"].append(teacher_acc_all)
    teacher_acc_train = (y_train == (yw_train if yw_train.ndim == 2 else yw_train.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc_train"].append(teacher_acc_train)
    teacher_acc_val = (y_val == yw_val).float().mean()
    results["teacher_acc_val"].append(teacher_acc_val)
    teacher_acc_test = (y_test == yw_test).float().mean()
    results["teacher_acc_test"].append(teacher_acc_test)

    if type(teacher_acc) == float:
        teacher_acc = torch.tensor([teacher_acc], device=cfg["device"])
    logger.info(f"Teacher label accuracy (all data, not combined): {[np.round(tacc.item() if hasattr(tacc, 'item') else tacc, 4) for tacc in teacher_acc]}")
    logger.info(f"Teacher label accuracy (all data): {teacher_acc_all:.4f}")
    logger.info(f"Teacher label accuracy (train): {teacher_acc_train:.4f}")
    logger.info(f"Teacher label accuracy (val): {teacher_acc_val:.4f}")
    logger.info(f"Teacher label accuracy (test): {teacher_acc_test:.4f}")

    ### w2s with entropy hard sample curriculum
    if before_optim_run_callback_weak is not None:
        before_optim_run_callback_weak(yw=yw_train, sample_idxs=np.arange(len(yw_train)))
    seed_all(cfg["seed"])

    # Read hard + entropy curriculum params from cfg (with defaults)
    entropy_hard_curriculum_kwargs = {
        'warmup_hard_epochs': cfg["w2s"].get("warmup_hard_epochs", 5),
        'cluster_method': cfg["w2s"].get("cluster_method", "gmm"),
        'weak_conf_threshold': cfg["w2s"].get("weak_conf_threshold", 0.6),
        'w2sg_conf_threshold': cfg["w2s"].get("w2sg_conf_threshold", 0.6),
        'entropy_threshold': cfg["w2s"].get("entropy_threshold", 1.5),
        'd0_weight': cfg["w2s"].get("d0_weight", 1.0),
        'd1_weight': cfg["w2s"].get("d1_weight", 2.0),
        'd2_weight': cfg["w2s"].get("d2_weight", 1.5),
    }
    logger.info(f"Entropy hard curriculum params: {entropy_hard_curriculum_kwargs}")

    results_teacher_to_student, student_model_probe = train_logreg_entropy_hard(
        x_train, yw_train, eval_datasets, device=cfg["device"],
        batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["teacher_labels_loss_fn_name"]](**(cfg["w2s"]["teacher_labels_loss_fn_kwargs"] or dict())),
        n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None,
        before_batch_callback=before_batch_callback_weak,
        after_batch_callback=after_batch_callback_weak,
        **entropy_hard_curriculum_kwargs,
    )
    results["results_teacher_to_student"].append(results_teacher_to_student)
    results["student_model_probe"].append(student_model_probe)

    ### gt
    if before_optim_run_callback_gt is not None:
        before_optim_run_callback_gt(yw=yw_test, sample_idxs=len(y_train) + len(y_val) + np.arange(len(yw_test)))
    seed_all(cfg["seed"])
    results_gt, gt_model_probe = train_logreg(x_train, y_train, eval_datasets, device=cfg["device"], batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["gt_labels_loss_fn_name"]](**(cfg["w2s"]["gt_labels_loss_fn_kwargs"] or dict())), n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None, before_batch_callback=before_batch_callback_gt, after_batch_callback=after_batch_callback_gt)
    results["results_gt"].append(results_gt)

    ### W2SG visualization (toggleable) — after both w2sg and gt training
    _balance_stats = None
    if '_balance_before' in results_teacher_to_student and '_balance_after' in results_teacher_to_student:
        _balance_stats = {
            'before': results_teacher_to_student['_balance_before'],
            'after': results_teacher_to_student['_balance_after'],
        }
    if cfg["w2s"].get("plot_w2sg", False):
        _plot_dir = cfg["w2s"].get("plot_save_dir", None)
        if _plot_dir:
            _dim_methods = cfg["w2s"].get("plot_dim_reduction", ["pca"])
            if isinstance(_dim_methods, str):
                _dim_methods = [_dim_methods]
            for _dm in _dim_methods:
                try:
                    plot_w2sg_analysis(
                        model=student_model_probe,
                        gt_model=gt_model_probe,
                        eval_datasets=eval_datasets,
                        save_dir=_plot_dir,
                        seed=cfg["seed"],
                        device=cfg["device"],
                        dim_reduction=_dm,
                        balance_stats=_balance_stats,
                        logger=logger,
                    )
                except Exception as _plot_err:
                    logger.info(f"[W2SG Plot] Warning: plotting failed ({_dm}): {_plot_err}")

    if return_data:
        return results, student_model_probe, {"x": x, "y": y, "yw": yw, "x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val, "x_test": x_test, "y_test": y_test, "yw_train": yw_train, "yw_val": yw_val, "yw_test": yw_test}
    return results, student_model_probe



def train_head_DG_selfMix(
    teacher_model,
    student_model,
    val_dataloader,     # CHANGED: Thay dataloader bằng val_dataloader
    test_dataloader,    # NEW: Thêm test_dataloader
    cfg,
    logger,
    cached_labels_path,
    cached_embs_path,
    results,
    rng,
    n_classes,
    return_data=False,
    additional_eval_data=None,
    before_optim_run_callback_weak=None,
    before_optim_run_callback_gt=None,
    after_batch_callback_weak=None,
    before_batch_callback_weak=None,
    after_batch_callback_gt=None,
    before_batch_callback_gt=None,
):
    """
    Thiết kế đầu vào nhận 2 domain val và test
    chia val thành 0.8, 0.2: train, val cho w2s
    test: target data lấy toàn bộ
    """
    
    ### get (weak) labels from current teacher
    if cfg["w2s"]["load_labels"] and os.path.exists(cached_labels_path):
        # load from cache
        logger.info("Loading teacher labels from cache...")
        cached = torch.load(cached_labels_path, pickle_module=dill, map_location="cpu")
        val_gt_labels, val_teacher_labels = cached["val_gt_labels"], cached["val_teacher_labels"]
        test_gt_labels, test_teacher_labels = cached["test_gt_labels"], cached["test_teacher_labels"]
        teacher_acc = cached.get("teacher_acc", 0) # Fallback if not saved
    else:
        # collect (and save) for BOTH val and test
        chunking_dir = os.path.join(os.path.dirname(cached_labels_path), f"label_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info("Collecting teacher labels for Validation Set...")
        _, val_gt_labels, val_teacher_labels, val_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=val_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'val'))
        
        logger.info("Collecting teacher labels for Test Set...")
        _, test_gt_labels, test_teacher_labels, test_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=test_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'test'))
        
        # Trích xuất giá trị số thực (float) an toàn trước khi tính trung bình
        val_acc_float = np.mean(val_teacher_acc)
        test_acc_float = np.mean(test_teacher_acc)
        teacher_acc = float((val_acc_float + test_acc_float) / 2.0)

        if cfg["w2s"]["save_labels"]:
            torch.save({
                "cfg": cfg,
                "val_gt_labels": val_gt_labels,
                "val_teacher_labels": val_teacher_labels,
                "test_gt_labels": test_gt_labels,
                "test_teacher_labels": test_teacher_labels,
                "teacher_acc": teacher_acc,
            }, cached_labels_path, pickle_module=dill)

    ### get embeddings from the student model
    if cfg["w2s"]["load_embeddings"] and os.path.exists(cached_embs_path):
        # load from cache
        logger.info("Loading student model embeddings from cache...")
        cached = torch.load(cached_embs_path, pickle_module=dill)
        val_student_embeddings, val_student_gt_labels = cached["val_embeddings"], cached["val_gt_labels"]
        test_student_embeddings, test_student_gt_labels = cached["test_embeddings"], cached["test_gt_labels"]
    else:
        # collect (and save) for BOTH val and test
        chunking_dir = os.path.join(os.path.dirname(cached_embs_path), f"embs_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        
        logger.info(f"Collecting student embeddings for Validation Set...")
        val_student_embeddings, val_student_gt_labels, val_student_labels, _, _, _ = preload(model=student_model, loader=val_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'val'), store_embs=True)
        #breakpoint()
        logger.info(f"Collecting student embeddings for Test Set...")
        test_student_embeddings, test_student_gt_labels, test_student_labels, _, _, _ = preload(model=student_model, loader=test_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'test'), store_embs=True)

        if cfg["w2s"]["save_embeddings"]:
            torch.save({
                "cfg": cfg,
                "val_embeddings": val_student_embeddings,
                "val_gt_labels": val_student_gt_labels,
                "test_embeddings": test_student_embeddings,
                "test_gt_labels": test_student_gt_labels,
            }, cached_embs_path, pickle_module=dill)
    # breakpoint()
    assert torch.all(val_gt_labels == val_student_gt_labels), "Val GT labels from teacher and student do not match."
    # breakpoint()
    assert torch.all(test_gt_labels == test_student_gt_labels), "Test GT labels from teacher and student do not match."
    del val_student_gt_labels, test_student_gt_labels

    ### CHANGED: Trộn (shuffle) và cắt dữ liệu chỉ cho tập VAL
    order = np.arange(len(val_gt_labels))
    rng.shuffle(order)
    results["order"].append(order)

    x_val_all = val_student_embeddings[order]
    y_val_all = val_gt_labels[order]
    yw_val_all = val_teacher_labels[order]

    ### split Validation into Train & Val for w2s
    assert len(cfg["w2s"]["train_val_test_split_DG"]) == 2, "Train/val split config must be of length 2 (e.g., [0.8, 0.2])."
    assert sum(cfg["w2s"]["train_val_test_split_DG"]) == 1.0, "Train/val split must sum to 1."
    
    n_train = int(cfg["w2s"]["train_val_test_split_DG"][0] * len(x_val_all))
    
    # Lấy Train & Val từ val_dataloader
    x_train, x_val = x_val_all[:n_train], x_val_all[n_train:]
    y_train, y_val = y_val_all[:n_train], y_val_all[n_train:]
    yw_train, yw_val = yw_val_all[:n_train], yw_val_all[n_train:]

    # Lấy Test toàn bộ từ test_dataloader
    x_test = test_student_embeddings
    y_test = test_gt_labels
    yw_test = test_teacher_labels

    # Nối lại để tính toán logging chung
    x = torch.cat([x_train, x_val, x_test]) if isinstance(x_train, torch.Tensor) else np.concatenate([x_train, x_val, x_test])
    y = torch.cat([y_train, y_val, y_test]) if isinstance(y_train, torch.Tensor) else np.concatenate([y_train, y_val, y_test])
    yw = torch.cat([yw_train, yw_val, yw_test]) if isinstance(yw_train, torch.Tensor) else np.concatenate([yw_train, yw_val, yw_test])

    yw_val_raw = yw_val.mean(1) if yw_val.ndim == 3 else yw_val
    yw_val = yw_val_raw.argmax(-1) # only for evaluation
    yw_test_raw = yw_test.mean(1) if yw_test.ndim == 3 else yw_test
    yw_test = yw_test_raw.argmax(-1) # only for evaluation
    
    eval_datasets = {"val": (x_val, y_val), "val_weak": (x_val, yw_val), "val_weak_raw": (x_val, yw_val_raw), "test": (x_test, y_test), "test_weak": (x_test, yw_test), "test_weak_raw": (x_test, yw_test_raw)}

    if additional_eval_data is not None:
        for k, v in additional_eval_data.items():
            eval_datasets[k] = v
            
    logger.info(f"\nTotal number of samples: {len(x)}.")
    logger.info(f"  Number of training samples (from Val Dataloader): {len(x_train)}.")
    logger.info(f"  Number of validation samples (from Val Dataloader): {len(x_val)}.")
    logger.info(f"  Number of testing samples (from Test Dataloader): {len(x_test)}.")

    ### eval teacher (average weak labels)
    results["teacher_acc_src"].append(teacher_acc)
    teacher_acc_all = (y == (yw if yw.ndim == 2 else yw.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc"].append(teacher_acc_all)
    teacher_acc_train = (y_train == (yw_train if yw_train.ndim == 2 else yw_train.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc_train"].append(teacher_acc_train)
    teacher_acc_val = (y_val == yw_val).float().mean()
    results["teacher_acc_val"].append(teacher_acc_val)
    teacher_acc_test = (y_test == yw_test).float().mean()
    results["teacher_acc_test"].append(teacher_acc_test)
    
    if type(teacher_acc) == float:
        teacher_acc = torch.tensor([teacher_acc], device=cfg["device"])
    logger.info(f"Teacher label accuracy (all data, not combined): {[np.round(tacc.item() if hasattr(tacc, 'item') else tacc, 4) for tacc in teacher_acc]}")
    logger.info(f"Teacher label accuracy (all data): {teacher_acc_all:.4f}")
    logger.info(f"Teacher label accuracy (train): {teacher_acc_train:.4f}")
    logger.info(f"Teacher label accuracy (val): {teacher_acc_val:.4f}")
    logger.info(f"Teacher label accuracy (test): {teacher_acc_test:.4f}")
    '''
    ### [NEW] eval student (Zero-shot / Pre-trained accuracy)
    # Nếu bạn có sẵn hàm model (chưa fine-tune), bạn có thể đánh giá trực tiếp trên embeddings
    # logger.info("--- Evaluating Base Student Performance ---")
    
    # student_model.eval()
    # with torch.no_grad():
    #     device = cfg["device"]
        
    #     # Tạo hàm đánh giá nhanh
    #     def eval_student(features, labels):
    #         features = torch.tensor(features, device=device) if not isinstance(features, torch.Tensor) else features.to(device)
    #         labels = torch.tensor(labels, device=device) if not isinstance(labels, torch.Tensor) else labels.to(device)
            
    #         # Giả định student_model có thể nhận features trực tiếp
    #         # NẾU student_model nhận ảnh thô (không phải features), phần này sẽ cần sửa lại để gọi qua DataLoader
    #         try:
    #             logits = student_model(features)
    #             if logits.ndim > 2: logits = logits.mean(1)
    #             preds = torch.argmax(logits, dim=-1)
                
    #             if labels.ndim > 1: labels = torch.argmax(labels, dim=-1)
    #             acc = (preds == labels).float().mean().item()
    #             return acc
    #         except Exception as e:
    #             return f"N/A (Error: {str(e)})"
        
    #     student_acc_train = eval_student(x_train, y_train)
    #     student_acc_val = eval_student(x_val, y_val)
    #     student_acc_test = eval_student(x_test, y_test)
        
    #     logger.info(f"Base Student label accuracy (train): {student_acc_train}")
    #     logger.info(f"Base Student label accuracy (val): {student_acc_val}")
    #     logger.info(f"Base Student label accuracy (test): {student_acc_test}")
    
    ### [NEW] eval student (Zero-shot / Pre-trained accuracy)
    # logger.info("--- Evaluating Base Student Performance on Raw Data ---")
    
    # def eval_base_model(model, dataloader, device):
    #     model.eval()
    #     correct, total = 0, 0
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             # Dataloader thường trả về tuple (x, y, sample_idxs, ...)
    #             x = batch[0].to(device)
    #             y = batch[1].to(device)
                
    #             try:
    #                 outputs = model(x)
    #                 # Xử lý trường hợp model trả về Tuple (VD: HuggingFace model trả về loss, logits)
    #                 logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        
    #                 if logits.ndim > 2: 
    #                     logits = logits.mean(1)
    #                 preds = torch.argmax(logits, dim=-1)
                    
    #                 if y.ndim > 1: 
    #                     y = torch.argmax(y, dim=-1)
                        
    #                 correct += (preds == y).float().sum().item()
    #                 total += len(y)
    #             except Exception as e:
    #                 return f"N/A (Error: {str(e)})"
                    
    #     return round(correct / total, 4) if total > 0 else 0

    # # Chạy đánh giá trực tiếp trên toàn bộ loader gốc
    # student_acc_val_all = eval_base_model(student_model, val_dataloader, cfg["device"])
    # student_acc_test_all = eval_base_model(student_model, test_dataloader, cfg["device"])
    
    # logger.info(f"Base Student label accuracy (Entire Validation Set): {student_acc_val_all}")
    # logger.info(f"Base Student label accuracy (Entire Test Set): {student_acc_test_all}")
    # logger.info("-------------------------------------------------------")
    '''
    ### w2s (Tiếp tục training...)

    ### w2s
    if before_optim_run_callback_weak is not None:
        before_optim_run_callback_weak(yw=yw_train, sample_idxs=np.arange(len(yw_train)))
    seed_all(cfg["seed"]) # important to get same results for cached/not cached
    # results_teacher_to_student, student_model_probe = train_logreg(x_train,
    #                                                                yw_train,
    #                                                                eval_datasets,
    #                                                                device=cfg["device"],
    #     batch_size=cfg["w2s"]["batch_size"],
    #     loss_fn=LOSS_DICT[cfg["w2s"]["teacher_labels_loss_fn_name"]](**(cfg["w2s"]["teacher_labels_loss_fn_kwargs"] or dict())),
    #     n_epochs=cfg["w2s"]["n_epochs"],
    #     lr=cfg["w2s"]["lr"],
    #     n_classes=n_classes,
    #     sample_weights=None,
    #     before_batch_callback=before_batch_callback_weak,
    #     after_batch_callback=after_batch_callback_weak)
    # results["results_teacher_to_student"].append(results_teacher_to_student)
    # results["student_model_probe"].append(student_model_probe)
    
    # --------------------------------------------------------------------------
    # THAY ĐỔI TẠI ĐÂY: Sử dụng train_selfmix_probe thay vì train_logreg
    # --------------------------------------------------------------------------
    logger.info("Training Student Head with SelfMix (GMM + Manifold Mixup)...")
    results_teacher_to_student, student_model_probe = train_selfmix_probe(
        x_train=x_train, 
        y_train=yw_train, 
        eval_datasets=eval_datasets, 
        device=cfg["device"],
        loss_fn=None,
        n_classes=n_classes,
        batch_size=cfg["w2s"].get("batch_size", 256),
        n_epochs=cfg["w2s"].get("n_epochs", 50),
        lr=cfg["w2s"].get("lr", 1e-3),
        alpha=4.0,       # Hệ số của phân phối Beta dùng cho Mixup
        lambda_p=1.0,  # Trọng số cho Pseudo-Loss (Eq. 17)
        lambda_r=5.0,  # Trọng số cho R-Drop Loss (Eq. 17)
        T=0.5,           # Nhiệt độ để làm sắc nét (sharpening) pseudo-labels
        warmup_epochs=6
    )
    
    # Do hàm trả về dict thay vì tuple thuần của train_logreg, ta bọc lại format cho khớp results dictionary
    results["results_teacher_to_student"].append(results_teacher_to_student)
    results["student_model_probe"].append(student_model_probe)
    # --------------------------------------------------------------------------

    ### gt
    if before_optim_run_callback_gt is not None:
        # CHANGED: Index của tập test bị dời đi một đoạn bằng tổng số mẫu của train và val
        before_optim_run_callback_gt(yw=yw_test, sample_idxs=len(y_train) + len(y_val) + np.arange(len(yw_test)))
    seed_all(cfg["seed"])
    results_gt, _ = train_logreg(x_train, y_train, eval_datasets, device=cfg["device"], batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["gt_labels_loss_fn_name"]](**(cfg["w2s"]["gt_labels_loss_fn_kwargs"] or dict())), n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None, before_batch_callback=before_batch_callback_gt, after_batch_callback=after_batch_callback_gt)
    results["results_gt"].append(results_gt)

    if return_data:
        return results, student_model_probe, {"x": x, "y": y, "yw": yw, "x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val, "x_test": x_test, "y_test": y_test, "yw_train": yw_train, "yw_val": yw_val, "yw_test": yw_test}
    return results, student_model_probe


def train_logreg(
    x_train,
    y_train,
    eval_datasets,
    device,
    loss_fn,
    n_epochs=20,
    weight_decay=0.0,
    lr=1e-3,
    batch_size=128,
    n_classes=1000,
    sample_weights=None,
    before_batch_callback=None,
    after_batch_callback=None,
):
    ### setup training data
    x_train = x_train.float()
    train_ds = torch.utils.data.TensorDataset(
        x_train,
        y_train,
        torch.arange(len(y_train)),
        sample_weights if sample_weights is not None else torch.ones(len(y_train), device=device),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    ### setup model and optimizer
    model = torch.nn.Linear(x_train.shape[-1], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    iter_i = 0
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)
    warning_printed = False

    ### train and eval
    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):
        ### train
        correct, total = 0, 0
        for b_i, (x, y, sample_idxs, sample_ws) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            if before_batch_callback is not None:
                x, y, pred, sample_ws = before_batch_callback(x=x, y=y, pred=pred, sample_idxs=sample_idxs, sample_ws=sample_ws, epoch=epoch, is_eval=False)
            if pred.ndim > 2:
                if not warning_printed:
                    print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                    warning_printed = True
                pred = pred.mean(1)
            loss = loss_fn(pred, y, step_frac=(iter_i := iter_i + 1) / n_iter, sample_weights=sample_ws)
            loss.backward()
            optimizer.step()
            schedule.step()

            ### calc metrics
            if len(y.shape) == 2:
                y = torch.argmax(y, dim=-1)
            elif len(y.shape) == 3:
                y = y.mean(1).argmax(-1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
            if after_batch_callback is not None:
                after_batch_callback(x=x, y=y, pred=pred, loss=loss, last_in_epoch=b_i == n_batches - 1, epoch=epoch)
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f}")

        ### eval
        with torch.no_grad():
            for key, (x_test, y_test) in eval_datasets.items():
                x_test = x_test.float().to(device)
                pred = model(x_test).detach().cpu()
                if pred.ndim > 2:
                    if not warning_printed:
                        print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                        warning_printed = True
                    pred = pred.mean(1)
                if len(pred.shape) > 1:
                    pred = torch.argmax(pred, dim=-1)
                if len(y_test.shape) > 1:
                    y_test = torch.argmax(y_test, dim=-1)
                # Ensure device match
                y_test = y_test.to(pred.device)
                acc = (pred == y_test).float().mean()
                results[f"{key}_all"].append(acc)

        ### print w2sg vs weak agreement/disagreement on test set every 2 epochs
        if (epoch + 1) % 5 == 0 and "test" in eval_datasets and "test_weak" in eval_datasets:
            with torch.no_grad():
                x_t, y_t = eval_datasets["val"]
                _, yw_t = eval_datasets["val_weak"]  # yw_t are weak hard labels
                weak_raw_available = "val_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["val_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = model(x_t).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                # Compute softmax probabilities and entropy: H = -sum(p * log(p))
                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)  # add eps to avoid log(0)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)  # shape: (n_test,)

                pred_t = torch.argmax(logits_t, dim=-1)
                
                # We need to compute weak confidence
                if weak_raw_available:
                    # yw_t_raw might be logits or probabilities
                    # To be safe, if min >= 0 and max <= 1.01 it's likely probs, else logits
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t
                
                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                # Masks for 4 cases
                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                # Average entropy for each case
                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                # Average confidence (max softmax probability) for each case
                confidence_t = probs_t.max(dim=-1).values  # shape: (n_test,)
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')
                
                # Average weak confidence for each case
                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Train w2sg set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")

            with torch.no_grad():
                x_t, y_t = eval_datasets["test"]
                _, yw_t = eval_datasets["test_weak"]  # yw_t are weak hard labels
                weak_raw_available = "test_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["test_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = model(x_t).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                # Compute softmax probabilities and entropy: H = -sum(p * log(p))
                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)  # add eps to avoid log(0)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)  # shape: (n_test,)

                pred_t = torch.argmax(logits_t, dim=-1)
                
                # We need to compute weak confidence
                if weak_raw_available:
                    # yw_t_raw might be logits or probabilities
                    # To be safe, if min >= 0 and max <= 1.01 it's likely probs, else logits
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t
                
                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                # Masks for 4 cases
                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                # Average entropy for each case
                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                # Average confidence (max softmax probability) for each case
                confidence_t = probs_t.max(dim=-1).values  # shape: (n_test,)
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')
                
                # Average weak confidence for each case
                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Test set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")

    ### final results
    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]

    return results, model




# =============================================================================
# Clustering helper for confidence-based data splitting
# =============================================================================
def cluster_by_confidence(confidence_values, method='gmm', threshold=0.5):
    """
    Cluster 1D confidence values into 2 groups (high vs low confidence).

    Args:
        confidence_values: numpy array or torch tensor of shape (N,)
        method: 'gmm', 'kmeans', 'threshold', 'median'
        threshold: GMM/KMeans probability threshold or direct threshold value

    Returns:
        high_conf_mask: boolean numpy array (N,), True = high confidence
        cluster_info: dict with clustering details
    """
    if isinstance(confidence_values, torch.Tensor):
        confidence_values = confidence_values.cpu().numpy()
    confidence_values = confidence_values.flatten().astype(np.float64)

    conf_2d = confidence_values.reshape(-1, 1)

    if method == 'gmm':
        from sklearn.mixture import GaussianMixture
        conf_range = confidence_values.max() - confidence_values.min()
        if conf_range == 0:
            return np.ones(len(confidence_values), dtype=bool), {'method': 'gmm', 'note': 'all identical'}
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=5e-4)
        gmm.fit(conf_2d)
        probs = gmm.predict_proba(conf_2d)
        high_cluster = int(gmm.means_.argmax())
        high_conf_mask = probs[:, high_cluster] >= threshold
        cluster_info = {
            'method': 'gmm',
            'high_cluster_mean': float(gmm.means_[high_cluster, 0]),
            'low_cluster_mean': float(gmm.means_[1 - high_cluster, 0]),
        }

    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        conf_range = confidence_values.max() - confidence_values.min()
        if conf_range == 0:
            return np.ones(len(confidence_values), dtype=bool), {'method': 'kmeans', 'note': 'all identical'}
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(conf_2d)
        high_cluster = int(km.cluster_centers_.argmax())
        high_conf_mask = (labels == high_cluster)
        cluster_info = {
            'method': 'kmeans',
            'high_cluster_center': float(km.cluster_centers_[high_cluster, 0]),
            'low_cluster_center': float(km.cluster_centers_[1 - high_cluster, 0]),
        }

    elif method == 'threshold':
        high_conf_mask = (confidence_values >= threshold)
        cluster_info = {
            'method': 'threshold',
            'threshold': float(threshold),
        }

    elif method == 'median':
        median_val = float(np.median(confidence_values))
        high_conf_mask = (confidence_values >= median_val)
        cluster_info = {
            'method': 'median',
            'median': median_val,
        }

    else:
        raise ValueError(f"Unknown clustering method: {method}. Choose from 'gmm', 'kmeans', 'threshold', 'median'.")

    # Safety: ensure at least some samples are kept
    min_samples = max(10, int(0.01 * len(confidence_values)))
    if high_conf_mask.sum() < min_samples:
        print(f"[WARNING] Only {high_conf_mask.sum()} samples in high-confidence cluster (min={min_samples}). Keeping all samples.")
        high_conf_mask = np.ones(len(confidence_values), dtype=bool)

    return high_conf_mask, cluster_info


# =============================================================================
# cluster_by_confidence_v2: Confidence filtering with per-class minimum guarantee
# =============================================================================
def cluster_by_confidence_v2(confidence_values, labels, n_classes, method='gmm', threshold=0.5, min_samples_per_class=5):
    """
    Cluster 1D confidence values into 2 groups (high vs low confidence),
    then ensure every class has at least `min_samples_per_class` samples.

    If after filtering a class has fewer than `min_samples_per_class` samples,
    additional samples of that class are pulled back from the rejected pool,
    selecting those with the highest confidence first.

    Args:
        confidence_values: numpy array or torch tensor of shape (N,)
        labels: torch tensor of shape (N,), (N, C), or (N, K, C) — training labels
                (soft or hard). Used to determine class membership.
        n_classes: int, total number of classes
        method: 'gmm', 'kmeans', 'threshold', 'median'
        threshold: GMM/KMeans probability threshold or direct threshold value
        min_samples_per_class: int, minimum samples per class after filtering (default 5)

    Returns:
        keep_mask: boolean numpy array (N,), True = keep this sample
        cluster_info: dict with clustering details (includes backfill info)
    """
    # --- Step 1: Run the original clustering ---
    high_conf_mask, cluster_info = cluster_by_confidence(
        confidence_values, method=method, threshold=threshold
    )

    # --- Step 2: Extract hard labels from (possibly soft) labels ---
    if isinstance(labels, torch.Tensor):
        if labels.ndim == 3:
            hard_labels = labels.float().mean(1).argmax(-1).cpu().numpy()
        elif labels.ndim == 2:
            hard_labels = labels.argmax(-1).cpu().numpy()
        else:
            hard_labels = labels.cpu().numpy()
    else:
        hard_labels = np.array(labels)
    hard_labels = hard_labels.astype(np.int64)

    # Ensure confidence_values is numpy
    if isinstance(confidence_values, torch.Tensor):
        conf_np = confidence_values.cpu().numpy().flatten().astype(np.float64)
    else:
        conf_np = np.array(confidence_values).flatten().astype(np.float64)

    # --- Step 3: Check per-class counts and backfill if needed ---
    keep_mask = high_conf_mask.copy()
    backfill_info = {}
    total_backfilled = 0

    for cls_id in range(n_classes):
        # Indices of all samples belonging to this class
        cls_all_indices = np.where(hard_labels == cls_id)[0]
        if len(cls_all_indices) == 0:
            # This class has zero samples in the entire dataset, nothing to backfill
            continue

        # Count how many are currently kept
        cls_kept_count = keep_mask[cls_all_indices].sum()

        if cls_kept_count < min_samples_per_class:
            # Find rejected samples of this class
            cls_rejected_indices = cls_all_indices[~keep_mask[cls_all_indices]]

            if len(cls_rejected_indices) == 0:
                # All samples of this class are already kept (but < min), nothing more to add
                continue

            # Number of additional samples needed
            n_needed = min_samples_per_class - int(cls_kept_count)

            # Sort rejected samples by confidence descending (pick highest confidence first)
            rejected_confs = conf_np[cls_rejected_indices]
            sorted_order = np.argsort(-rejected_confs)  # descending
            n_to_add = min(n_needed, len(cls_rejected_indices))

            # Add these samples back to the keep mask
            indices_to_add = cls_rejected_indices[sorted_order[:n_to_add]]
            keep_mask[indices_to_add] = True

            backfill_info[int(cls_id)] = {
                'had': int(cls_kept_count),
                'added': n_to_add,
                'now': int(cls_kept_count) + n_to_add,
                'total_in_dataset': len(cls_all_indices),
            }
            total_backfilled += n_to_add

    # Update cluster_info with backfill details
    cluster_info['backfill'] = {
        'min_samples_per_class': min_samples_per_class,
        'total_backfilled': total_backfilled,
        'n_classes_backfilled': len(backfill_info),
        'per_class': backfill_info,
    }

    if total_backfilled > 0:
        print(f"  [cluster_by_confidence_v2] Backfilled {total_backfilled} samples across "
              f"{len(backfill_info)} classes (min_per_class={min_samples_per_class})")
        for cls_id, info in backfill_info.items():
            print(f"    Class {cls_id}: {info['had']} -> {info['now']} "
                  f"(+{info['added']}, total in dataset: {info['total_in_dataset']})")

    return keep_mask, cluster_info


# =============================================================================
# train_logreg_hard: Curriculum learning with confidence-based easy/hard split
# =============================================================================
def train_logreg_hard(
    x_train,
    y_train,
    eval_datasets,
    device,
    loss_fn,
    n_epochs=20,
    weight_decay=0.0,
    lr=1e-3,
    batch_size=128,
    n_classes=1000,
    sample_weights=None,
    before_batch_callback=None,
    after_batch_callback=None,
    # --- Hard sample curriculum params ---
    warmup_hard_epochs=3,
    hard_weight=2.0,
    cluster_method='gmm',        # 'gmm', 'kmeans', 'threshold', 'median'
    weak_conf_threshold=0.5,
    w2sg_conf_threshold=0.5,
):
    """
    train_logreg with hard sample curriculum learning:
      1. Before epoch 0: filter data by weak model confidence clustering
      2. Epochs 0 ~ warmup_hard_epochs-1: train on filtered data (equal weights)
      3. From warmup_hard_epochs: split filtered data into easy/hard by w2sg
         confidence, train with higher weight for hard samples
    """

    # =================================================================
    # PHASE 0: Compute weak confidence and filter training data
    # =================================================================
    print(f"\n{'='*60}")
    print(f"[Hard Curriculum] Phase 0: Filtering by weak confidence")
    print(f"  Cluster method: {cluster_method}, threshold: {weak_conf_threshold}")
    print(f"{'='*60}")

    # Compute weak confidence from y_train (= weak labels)
    if y_train.ndim == 3:
        yw_soft = y_train.float().mean(1)  # (N, K, C) -> (N, C)
    else:
        yw_soft = y_train.float()  # (N, C)

    # Check if probabilities or logits
    if yw_soft.min() >= 0 and yw_soft.max() <= 1.01:
        weak_probs = yw_soft
    else:
        weak_probs = torch.softmax(yw_soft, dim=-1)

    weak_confidence = weak_probs.max(dim=-1).values.cpu().numpy()

    # Cluster and filter (v2: ensures min samples per class)
    keep_mask, weak_cluster_info = cluster_by_confidence_v2(
        weak_confidence, labels=y_train, n_classes=n_classes,
        method=cluster_method, threshold=weak_conf_threshold,
        min_samples_per_class=5
    )

    n_before = len(x_train)
    x_train_filtered = x_train[keep_mask]
    y_train_filtered = y_train[keep_mask]
    n_after = len(x_train_filtered)

    # --- Class balance stats BEFORE filtering ---
    balance_before = compute_class_balance_stats(y_train, n_classes)
    print(f"  === Class Balance BEFORE filtering ===")
    print(format_balance_stats(balance_before, prefix="    "))

    print(f"  Samples before filtering: {n_before}")
    print(f"  Samples after filtering:  {n_after} ({n_after/n_before*100:.1f}%)")
    print(f"  Removed: {n_before - n_after} samples ({(n_before - n_after)/n_before*100:.1f}%)")
    print(f"  Weak confidence stats: min={weak_confidence.min():.4f}, max={weak_confidence.max():.4f}, mean={weak_confidence.mean():.4f}")
    print(f"  Cluster info: {weak_cluster_info}")

    # --- Class balance stats AFTER filtering ---
    balance_after = compute_class_balance_stats(y_train_filtered, n_classes)
    print(f"  === Class Balance AFTER filtering ===")
    print(format_balance_stats(balance_after, prefix="    "))

    # =================================================================
    # Setup training on filtered data
    # =================================================================
    x_train_filtered = x_train_filtered.float()
    weights_tensor = torch.ones(n_after, device=device)

    train_ds = torch.utils.data.TensorDataset(
        x_train_filtered,
        y_train_filtered,
        torch.arange(n_after),
        weights_tensor,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    ### setup model and optimizer
    model = torch.nn.Linear(x_train_filtered.shape[-1], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    iter_i = 0
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)
    warning_printed = False

    ### train and eval
    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    results["hard_curriculum_info"] = []

    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):

        # =============================================================
        # PHASE 1 & 2: Update sample weights based on epoch
        # =============================================================
        if epoch >= warmup_hard_epochs:
            # ---- PHASE 2: Cluster by w2sg confidence ----
            model.eval()
            with torch.no_grad():
                logits = model(x_train_filtered.to(device))
                w2sg_probs = torch.softmax(logits, dim=-1)
                w2sg_confidence = w2sg_probs.max(dim=-1).values.cpu().numpy()

            # Cluster: high confidence = easy, low confidence = hard
            easy_mask, w2sg_cluster_info = cluster_by_confidence(
                w2sg_confidence, method=cluster_method, threshold=w2sg_conf_threshold
            )
            hard_mask = ~easy_mask

            # Update weights in-place
            easy_indices = torch.from_numpy(easy_mask).bool()
            hard_indices = ~easy_indices
            weights_tensor.fill_(1.0)
            weights_tensor[hard_indices] = hard_weight

            n_easy = int(easy_mask.sum())
            n_hard = int(hard_mask.sum())

            curriculum_info = {
                'epoch': epoch,
                'phase': 'hard_curriculum',
                'n_easy': n_easy,
                'n_hard': n_hard,
                'easy_pct': float(n_easy / n_after * 100),
                'hard_pct': float(n_hard / n_after * 100),
                'w2sg_cluster_info': w2sg_cluster_info,
            }
            results["hard_curriculum_info"].append(curriculum_info)

            if epoch == warmup_hard_epochs or (epoch + 1) % 5 == 0:
                print(f"\n  [Epoch {epoch}] Hard Curriculum Split ({cluster_method}):")
                print(f"    Easy: {n_easy} ({n_easy/n_after*100:.1f}%) | Hard: {n_hard} ({n_hard/n_after*100:.1f}%)")
                print(f"    Hard weight: {hard_weight}")
                print(f"    W2SG cluster info: {w2sg_cluster_info}")
        else:
            # ---- PHASE 1: Warmup with equal weights ----
            weights_tensor.fill_(1.0)
            n_easy, n_hard = n_after, 0
            results["hard_curriculum_info"].append({
                'epoch': epoch, 'phase': 'warmup', 'n_samples': n_after
            })

        # =============================================================
        # Training loop
        # =============================================================
        model.train()
        correct, total = 0, 0
        for b_i, (x, y, sample_idxs, sample_ws) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            if before_batch_callback is not None:
                x, y, pred, sample_ws = before_batch_callback(x=x, y=y, pred=pred, sample_idxs=sample_idxs, sample_ws=sample_ws, epoch=epoch, is_eval=False)
            if pred.ndim > 2:
                if not warning_printed:
                    print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                    warning_printed = True
                pred = pred.mean(1)
            loss = loss_fn(pred, y, step_frac=(iter_i := iter_i + 1) / n_iter, sample_weights=sample_ws)
            loss.backward()
            optimizer.step()
            schedule.step()

            ### calc metrics
            if len(y.shape) == 2:
                y = torch.argmax(y, dim=-1)
            elif len(y.shape) == 3:
                y = y.mean(1).argmax(-1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
            if after_batch_callback is not None:
                after_batch_callback(x=x, y=y, pred=pred, loss=loss, last_in_epoch=b_i == n_batches - 1, epoch=epoch)

        phase_str = "warmup" if epoch < warmup_hard_epochs else f"E={n_easy}/H={n_hard}"
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f} [{phase_str}]")

        # =============================================================
        # Eval
        # =============================================================
        with torch.no_grad():
            for key, (x_test, y_test) in eval_datasets.items():
                x_test = x_test.float().to(device)
                pred = model(x_test).detach().cpu()
                if pred.ndim > 2:
                    if not warning_printed:
                        print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                        warning_printed = True
                    pred = pred.mean(1)
                if len(pred.shape) > 1:
                    pred = torch.argmax(pred, dim=-1)
                if len(y_test.shape) > 1:
                    y_test = torch.argmax(y_test, dim=-1)
                # Ensure device match
                y_test = y_test.to(pred.device)
                acc = (pred == y_test).float().mean()
                results[f"{key}_all"].append(acc)

        ### print w2sg vs weak agreement/disagreement on test set every 5 epochs
        if (epoch + 1) % 5 == 0 and "test" in eval_datasets and "test_weak" in eval_datasets:
            with torch.no_grad():
                x_t, y_t = eval_datasets["val"]
                _, yw_t = eval_datasets["val_weak"]  # yw_t are weak hard labels
                weak_raw_available = "val_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["val_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = model(x_t).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                # Compute softmax probabilities and entropy: H = -sum(p * log(p))
                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)  # add eps to avoid log(0)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)  # shape: (n_test,)

                pred_t = torch.argmax(logits_t, dim=-1)
                
                # We need to compute weak confidence
                if weak_raw_available:
                    # yw_t_raw might be logits or probabilities
                    # To be safe, if min >= 0 and max <= 1.01 it's likely probs, else logits
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t
                
                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                # Masks for 4 cases
                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                # Average entropy for each case
                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                # Average confidence (max softmax probability) for each case
                confidence_t = probs_t.max(dim=-1).values  # shape: (n_test,)
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')
                
                # Average weak confidence for each case
                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Train w2sg set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")

            with torch.no_grad():
                x_t, y_t = eval_datasets["test"]
                _, yw_t = eval_datasets["test_weak"]  # yw_t are weak hard labels
                weak_raw_available = "test_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["test_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = model(x_t).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                # Compute softmax probabilities and entropy
                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)

                pred_t = torch.argmax(logits_t, dim=-1)

                # Compute weak confidence
                if weak_raw_available:
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t

                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                # Masks for 4 cases
                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                # Average entropy for each case
                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                # Average confidence (max softmax probability) for each case
                confidence_t = probs_t.max(dim=-1).values
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                # Average weak confidence for each case
                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Test set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")

    ### final results
    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]

    # Store balance stats for visualization
    results['_balance_before'] = balance_before
    results['_balance_after'] = balance_after

    return results, model

# =============================================================================
# train_logreg_hard_v1: Curriculum learning with SAM optimizer
# (Sharpness-Aware Minimization for Efficiently Improving Generalization)
# Reference: Foret et al., ICLR 2021 — https://arxiv.org/abs/2010.01412
# =============================================================================
def train_logreg_hard_v1(
    x_train,
    y_train,
    eval_datasets,
    device,
    loss_fn,
    n_epochs=20,
    weight_decay=0.0,
    lr=1e-3,
    batch_size=128,
    n_classes=1000,
    sample_weights=None,
    before_batch_callback=None,
    after_batch_callback=None,
    # --- Hard sample curriculum params ---
    warmup_hard_epochs=3,
    hard_weight=2.0,
    cluster_method='gmm',        # 'gmm', 'kmeans', 'threshold', 'median'
    weak_conf_threshold=0.5,
    w2sg_conf_threshold=0.5,
    # --- SAM optimizer params ---
    sam_rho=0.05,                 # perturbation radius for SAM
    sam_adaptive=False,           # if True, use Adaptive SAM (ASAM)
):
    """
    train_logreg_hard variant using SAM (Sharpness-Aware Minimization).

    Identical pipeline to train_logreg_hard:
      Phase 0: Filter by weak model confidence (cluster_by_confidence_v2).
      Phase 1 (epochs < warmup_hard_epochs): Warmup with equal weights.
      Phase 2 (epochs >= warmup_hard_epochs): Split into easy/hard by w2sg
        confidence, upweight hard samples.

    Key difference: the linear classifier is optimized with SAM-wrapped Adam
    instead of plain Adam. SAM performs two forward-backward passes per batch:
      1. Compute loss & gradients at current weights -> perturb to worst-case point
      2. Compute loss & gradients at perturbed point -> update original weights
    This encourages convergence to flatter minima, improving generalization,
    which is particularly beneficial when training with noisy weak labels.

    Extra args (compared to train_logreg_hard):
      sam_rho (float): perturbation neighborhood radius (default: 0.05)
      sam_adaptive (bool): use ASAM scale-invariant perturbation (default: False)
    """
    from rw2s.losses import SAM

    # =================================================================
    # PHASE 0: Compute weak confidence and filter training data
    # =================================================================
    print(f"\n{'='*60}")
    print(f"[Hard Curriculum v1 + SAM] Phase 0: Filtering by weak confidence")
    print(f"  Cluster method: {cluster_method}, threshold: {weak_conf_threshold}")
    print(f"  SAM rho: {sam_rho}, adaptive: {sam_adaptive}")
    print(f"{'='*60}")

    # Compute weak confidence from y_train (= weak labels)
    if y_train.ndim == 3:
        yw_soft = y_train.float().mean(1)  # (N, K, C) -> (N, C)
    else:
        yw_soft = y_train.float()  # (N, C)

    # Check if probabilities or logits
    if yw_soft.min() >= 0 and yw_soft.max() <= 1.01:
        weak_probs = yw_soft
    else:
        weak_probs = torch.softmax(yw_soft, dim=-1)

    weak_confidence = weak_probs.max(dim=-1).values.cpu().numpy()

    # Cluster and filter (v2: ensures min samples per class)
    keep_mask, weak_cluster_info = cluster_by_confidence_v2(
        weak_confidence, labels=y_train, n_classes=n_classes,
        method=cluster_method, threshold=weak_conf_threshold,
        min_samples_per_class=5
    )

    n_before = len(x_train)
    x_train_filtered = x_train[keep_mask]
    y_train_filtered = y_train[keep_mask]
    n_after = len(x_train_filtered)

    # --- Class balance stats BEFORE filtering ---
    balance_before = compute_class_balance_stats(y_train, n_classes)
    print(f"  === Class Balance BEFORE filtering ===")
    print(format_balance_stats(balance_before, prefix="    "))

    print(f"  Samples before filtering: {n_before}")
    print(f"  Samples after filtering:  {n_after} ({n_after/n_before*100:.1f}%)")
    print(f"  Removed: {n_before - n_after} samples ({(n_before - n_after)/n_before*100:.1f}%)")
    print(f"  Weak confidence stats: min={weak_confidence.min():.4f}, max={weak_confidence.max():.4f}, mean={weak_confidence.mean():.4f}")
    print(f"  Cluster info: {weak_cluster_info}")

    # --- Class balance stats AFTER filtering ---
    balance_after = compute_class_balance_stats(y_train_filtered, n_classes)
    print(f"  === Class Balance AFTER filtering ===")
    print(format_balance_stats(balance_after, prefix="    "))

    # =================================================================
    # Setup training on filtered data
    # =================================================================
    x_train_filtered = x_train_filtered.float()
    weights_tensor = torch.ones(n_after, device=device)

    train_ds = torch.utils.data.TensorDataset(
        x_train_filtered,
        y_train_filtered,
        torch.arange(n_after),
        weights_tensor,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    ### setup model and SAM optimizer
    model = torch.nn.Linear(x_train_filtered.shape[-1], n_classes).to(device)

    # SAM wraps the base optimizer (Adam)
    optimizer = SAM(
        model.parameters(),
        base_optimizer=torch.optim.Adam,
        rho=sam_rho,
        adaptive=sam_adaptive,
        lr=lr,
        weight_decay=weight_decay,
    )

    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    iter_i = 0
    # LR scheduler should be attached to the base_optimizer (as per SAM docs)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer.base_optimizer, T_max=n_iter
    )
    warning_printed = False

    ### train and eval
    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    results["hard_curriculum_info"] = []

    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):

        # =============================================================
        # PHASE 1 & 2: Update sample weights based on epoch
        # =============================================================
        if epoch >= warmup_hard_epochs:
            # ---- PHASE 2: Cluster by w2sg confidence ----
            model.eval()
            with torch.no_grad():
                logits = model(x_train_filtered.to(device))
                w2sg_probs = torch.softmax(logits, dim=-1)
                w2sg_confidence = w2sg_probs.max(dim=-1).values.cpu().numpy()

            # Cluster: high confidence = easy, low confidence = hard
            easy_mask, w2sg_cluster_info = cluster_by_confidence(
                w2sg_confidence, method=cluster_method, threshold=w2sg_conf_threshold
            )
            hard_mask = ~easy_mask

            # Update weights in-place
            easy_indices = torch.from_numpy(easy_mask).bool()
            hard_indices = ~easy_indices
            weights_tensor.fill_(1.0)
            weights_tensor[hard_indices] = hard_weight

            n_easy = int(easy_mask.sum())
            n_hard = int(hard_mask.sum())

            curriculum_info = {
                'epoch': epoch,
                'phase': 'hard_curriculum',
                'n_easy': n_easy,
                'n_hard': n_hard,
                'easy_pct': float(n_easy / n_after * 100),
                'hard_pct': float(n_hard / n_after * 100),
                'w2sg_cluster_info': w2sg_cluster_info,
            }
            results["hard_curriculum_info"].append(curriculum_info)

            if epoch == warmup_hard_epochs or (epoch + 1) % 5 == 0:
                print(f"\n  [Epoch {epoch}] Hard Curriculum Split ({cluster_method}):")
                print(f"    Easy: {n_easy} ({n_easy/n_after*100:.1f}%) | Hard: {n_hard} ({n_hard/n_after*100:.1f}%)")
                print(f"    Hard weight: {hard_weight}")
                print(f"    W2SG cluster info: {w2sg_cluster_info}")
        else:
            # ---- PHASE 1: Warmup with equal weights ----
            weights_tensor.fill_(1.0)
            n_easy, n_hard = n_after, 0
            results["hard_curriculum_info"].append({
                'epoch': epoch, 'phase': 'warmup', 'n_samples': n_after
            })

        # =============================================================
        # Training loop (SAM: two forward-backward passes per batch)
        # =============================================================
        model.train()
        correct, total = 0, 0
        for b_i, (x, y, sample_idxs, sample_ws) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # ---- SAM first forward-backward pass ----
            optimizer.zero_grad()
            pred = model(x)
            if before_batch_callback is not None:
                x, y, pred, sample_ws = before_batch_callback(
                    x=x, y=y, pred=pred, sample_idxs=sample_idxs,
                    sample_ws=sample_ws, epoch=epoch, is_eval=False
                )
            if pred.ndim > 2:
                if not warning_printed:
                    print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                    warning_printed = True
                pred = pred.mean(1)
            loss = loss_fn(pred, y, step_frac=(iter_i := iter_i + 1) / n_iter, sample_weights=sample_ws)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # ---- SAM second forward-backward pass ----
            # Detach x, y, sample_ws: before_batch_callback may return tensors
            # that are nodes in the first computation graph. After loss.backward()
            # frees that graph, reusing them without detach() would raise:
            # "Trying to backward through the graph a second time"
            x_2 = x.detach()
            y_2 = y.detach() if isinstance(y, torch.Tensor) else y
            sw_2 = sample_ws.detach() if isinstance(sample_ws, torch.Tensor) else sample_ws
            pred_2 = model(x_2)
            if pred_2.ndim > 2:
                pred_2 = pred_2.mean(1)
            loss_2 = loss_fn(pred_2, y_2, step_frac=iter_i / n_iter, sample_weights=sw_2)
            loss_2.backward()
            optimizer.second_step(zero_grad=True)

            schedule.step()

            ### calc metrics (use pred from the first pass for logging)
            if len(y.shape) == 2:
                y = torch.argmax(y, dim=-1)
            elif len(y.shape) == 3:
                y = y.mean(1).argmax(-1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
            if after_batch_callback is not None:
                after_batch_callback(x=x, y=y, pred=pred, loss=loss, last_in_epoch=b_i == n_batches - 1, epoch=epoch)

        phase_str = "warmup" if epoch < warmup_hard_epochs else f"E={n_easy}/H={n_hard}"
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f} [{phase_str}]")

        # =============================================================
        # Eval
        # =============================================================
        with torch.no_grad():
            for key, (x_test, y_test) in eval_datasets.items():
                x_test = x_test.float().to(device)
                pred = model(x_test).detach().cpu()
                if pred.ndim > 2:
                    if not warning_printed:
                        print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                        warning_printed = True
                    pred = pred.mean(1)
                if len(pred.shape) > 1:
                    pred = torch.argmax(pred, dim=-1)
                if len(y_test.shape) > 1:
                    y_test = torch.argmax(y_test, dim=-1)
                # Ensure device match
                y_test = y_test.to(pred.device)
                acc = (pred == y_test).float().mean()
                results[f"{key}_all"].append(acc)

        ### print w2sg vs weak agreement/disagreement on test set every 5 epochs
        if (epoch + 1) % 5 == 0 and "test" in eval_datasets and "test_weak" in eval_datasets:
            with torch.no_grad():
                x_t, y_t = eval_datasets["val"]
                _, yw_t = eval_datasets["val_weak"]  # yw_t are weak hard labels
                weak_raw_available = "val_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["val_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = model(x_t).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                # Compute softmax probabilities and entropy: H = -sum(p * log(p))
                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)  # add eps to avoid log(0)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)  # shape: (n_test,)

                pred_t = torch.argmax(logits_t, dim=-1)
                
                # We need to compute weak confidence
                if weak_raw_available:
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t
                
                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                # Masks for 4 cases
                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                # Average entropy for each case
                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                # Average confidence (max softmax probability) for each case
                confidence_t = probs_t.max(dim=-1).values  # shape: (n_test,)
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')
                
                # Average weak confidence for each case
                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Train w2sg set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")

            with torch.no_grad():
                x_t, y_t = eval_datasets["test"]
                _, yw_t = eval_datasets["test_weak"]  # yw_t are weak hard labels
                weak_raw_available = "test_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["test_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = model(x_t).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                # Compute softmax probabilities and entropy
                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)

                pred_t = torch.argmax(logits_t, dim=-1)

                # Compute weak confidence
                if weak_raw_available:
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t

                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                # Masks for 4 cases
                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                # Average entropy for each case
                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                # Average confidence (max softmax probability) for each case
                confidence_t = probs_t.max(dim=-1).values
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                # Average weak confidence for each case
                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Test set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")

    ### final results
    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]

    # Store balance stats for visualization
    results['_balance_before'] = balance_before
    results['_balance_after'] = balance_after

    return results, model


# =============================================================================
# train_logreg_hard_v2: Curriculum learning with cosine-NN Mixup relabeling
# =============================================================================
def train_logreg_hard_v2(
    x_train,
    y_train,
    eval_datasets,
    device,
    loss_fn,
    n_epochs=20,
    weight_decay=0.0,
    lr=1e-3,
    batch_size=128,
    n_classes=1000,
    sample_weights=None,
    before_batch_callback=None,
    after_batch_callback=None,
    # --- Hard sample curriculum params ---
    warmup_hard_epochs=3,
    hard_weight=2.0,
    cluster_method='gmm',
    weak_conf_threshold=0.5,
    w2sg_conf_threshold=0.5,
    # --- Mixup relabeling params ---
    mixup_alpha=0.5,
):
    """
    train_logreg_hard variant with cosine-NN Mixup relabeling (v2).

    Keeps the same overall pipeline as train_logreg_hard but modifies Phase 2:
      Phase 0: Filter by weak model confidence (identical to v1).
      Phase 1 (epochs < warmup_hard_epochs): Warmup with equal weights.
      Phase 2 (epochs >= warmup_hard_epochs):
        1. Cluster filtered data into HIGH / LOW confidence groups via w2sg.
        2. For each LOW-confidence embedding x, find its nearest neighbor a
           in the HIGH-confidence group using cosine similarity.
        3. Relabel: y' = lambda * y + (1 - lambda) * b, where b is the weak
           label of the nearest neighbor and lambda ~ Beta(mixup_alpha, mixup_alpha).
        4. Merge back and continue training.
    """

    # =================================================================
    # PHASE 0: Compute weak confidence and filter training data
    # =================================================================
    print(f"\n{'='*60}")
    print(f"[Hard Curriculum v2] Phase 0: Filtering by weak confidence")
    print(f"  Cluster method: {cluster_method}, threshold: {weak_conf_threshold}")
    print(f"  Mixup alpha: {mixup_alpha}")
    print(f"{'='*60}")

    if y_train.ndim == 3:
        yw_soft = y_train.float().mean(1)
    else:
        yw_soft = y_train.float()

    if yw_soft.min() >= 0 and yw_soft.max() <= 1.01:
        weak_probs = yw_soft
    else:
        weak_probs = torch.softmax(yw_soft, dim=-1)

    weak_confidence = weak_probs.max(dim=-1).values.cpu().numpy()

    keep_mask, weak_cluster_info = cluster_by_confidence_v2(
        weak_confidence, labels=y_train, n_classes=n_classes,
        method=cluster_method, threshold=weak_conf_threshold,
        min_samples_per_class=5
    )

    n_before = len(x_train)
    x_train_filtered = x_train[keep_mask]
    y_train_filtered = y_train[keep_mask]
    n_after = len(x_train_filtered)

    balance_before = compute_class_balance_stats(y_train, n_classes)
    print(f"  === Class Balance BEFORE filtering ===")
    print(format_balance_stats(balance_before, prefix="    "))

    print(f"  Samples before filtering: {n_before}")
    print(f"  Samples after filtering:  {n_after} ({n_after/n_before*100:.1f}%)")
    print(f"  Removed: {n_before - n_after} samples ({(n_before - n_after)/n_before*100:.1f}%)")
    print(f"  Weak confidence stats: min={weak_confidence.min():.4f}, max={weak_confidence.max():.4f}, mean={weak_confidence.mean():.4f}")
    print(f"  Cluster info: {weak_cluster_info}")

    balance_after = compute_class_balance_stats(y_train_filtered, n_classes)
    print(f"  === Class Balance AFTER filtering ===")
    print(format_balance_stats(balance_after, prefix="    "))

    # =================================================================
    # Setup training on filtered data
    # =================================================================
    x_train_filtered = x_train_filtered.float()
    # y_train_filtered will be modified in-place during Phase 2, so clone it
    y_train_working = y_train_filtered.clone().float()
    weights_tensor = torch.ones(n_after, device=device)

    train_ds = torch.utils.data.TensorDataset(
        x_train_filtered,
        y_train_working,
        torch.arange(n_after),
        weights_tensor,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    ### setup model and optimizer
    model = torch.nn.Linear(x_train_filtered.shape[-1], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    iter_i = 0
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)
    warning_printed = False

    ### train and eval
    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    results["hard_curriculum_info"] = []

    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):

        # =============================================================
        # PHASE 1 & 2: Update labels based on epoch
        # =============================================================
        if epoch >= warmup_hard_epochs:
            # ---- PHASE 2: Cosine-NN Mixup relabeling ----
            model.eval()
            with torch.no_grad():
                logits = model(x_train_filtered.to(device))
                w2sg_probs = torch.softmax(logits, dim=-1)
                w2sg_confidence = w2sg_probs.max(dim=-1).values.cpu().numpy()

            # Cluster: high confidence = easy, low confidence = hard
            easy_mask, w2sg_cluster_info = cluster_by_confidence(
                w2sg_confidence, method=cluster_method, threshold=w2sg_conf_threshold
            )
            hard_mask = ~easy_mask

            n_easy = int(easy_mask.sum())
            n_hard = int(hard_mask.sum())

            # --- Cosine-NN Mixup relabeling for LOW confidence samples ---
            if n_hard > 0 and n_easy > 0:
                easy_indices = np.where(easy_mask)[0]
                hard_indices = np.where(hard_mask)[0]

                x_high = x_train_filtered[easy_indices]  # (n_easy, D)
                x_low = x_train_filtered[hard_indices]    # (n_hard, D)

                # Get weak labels (use original filtered labels, not working copy)
                y_high = y_train_filtered[easy_indices].float()  # (n_easy, C)
                y_low = y_train_filtered[hard_indices].float()   # (n_hard, C)
                if y_high.ndim == 3:
                    y_high = y_high.mean(1)
                if y_low.ndim == 3:
                    y_low = y_low.mean(1)

                # Cosine similarity: normalize then matmul => (n_hard, n_easy)
                x_low_norm = torch.nn.functional.normalize(x_low, p=2, dim=-1)
                x_high_norm = torch.nn.functional.normalize(x_high, p=2, dim=-1)
                cos_sim = torch.mm(x_low_norm, x_high_norm.t())  # (n_hard, n_easy)

                # Find nearest neighbor in high-conf group for each low-conf sample
                nn_indices = cos_sim.argmax(dim=-1)  # (n_hard,)
                b = y_high[nn_indices]  # (n_hard, C) — weak labels of nearest neighbors

                # Mixup: y' = lambda * y + (1 - lambda) * b
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                y_prime = lam * y_low + (1 - lam) * b  # (n_hard, C)

                # Update the working labels for hard samples
                y_train_working[hard_indices] = y_prime

                # Rebuild dataset with updated labels
                train_ds = torch.utils.data.TensorDataset(
                    x_train_filtered,
                    y_train_working,
                    torch.arange(n_after),
                    weights_tensor,
                )
                train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

            curriculum_info = {
                'epoch': epoch,
                'phase': 'hard_curriculum_v2_mixup',
                'n_easy': n_easy,
                'n_hard': n_hard,
                'easy_pct': float(n_easy / n_after * 100),
                'hard_pct': float(n_hard / n_after * 100),
                'mixup_lambda': float(lam) if n_hard > 0 and n_easy > 0 else None,
                'w2sg_cluster_info': w2sg_cluster_info,
            }
            results["hard_curriculum_info"].append(curriculum_info)

            if epoch == warmup_hard_epochs or (epoch + 1) % 5 == 0:
                print(f"\n  [Epoch {epoch}] Hard Curriculum v2 - Cosine-NN Mixup ({cluster_method}):")
                print(f"    High-conf (easy): {n_easy} ({n_easy/n_after*100:.1f}%) | Low-conf (hard): {n_hard} ({n_hard/n_after*100:.1f}%)")
                if n_hard > 0 and n_easy > 0:
                    print(f"    Mixup lambda: {lam:.4f} (alpha={mixup_alpha})")
                print(f"    W2SG cluster info: {w2sg_cluster_info}")
        else:
            # ---- PHASE 1: Warmup with equal weights ----
            weights_tensor.fill_(1.0)
            n_easy, n_hard = n_after, 0
            results["hard_curriculum_info"].append({
                'epoch': epoch, 'phase': 'warmup', 'n_samples': n_after
            })

        # =============================================================
        # Training loop
        # =============================================================
        model.train()
        correct, total = 0, 0
        for b_i, (x, y, sample_idxs, sample_ws) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            if before_batch_callback is not None:
                x, y, pred, sample_ws = before_batch_callback(x=x, y=y, pred=pred, sample_idxs=sample_idxs, sample_ws=sample_ws, epoch=epoch, is_eval=False)
            if pred.ndim > 2:
                if not warning_printed:
                    print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                    warning_printed = True
                pred = pred.mean(1)
            loss = loss_fn(pred, y, step_frac=(iter_i := iter_i + 1) / n_iter, sample_weights=sample_ws)
            loss.backward()
            optimizer.step()
            schedule.step()

            ### calc metrics
            if len(y.shape) == 2:
                y = torch.argmax(y, dim=-1)
            elif len(y.shape) == 3:
                y = y.mean(1).argmax(-1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
            if after_batch_callback is not None:
                after_batch_callback(x=x, y=y, pred=pred, loss=loss, last_in_epoch=b_i == n_batches - 1, epoch=epoch)

        phase_str = "warmup" if epoch < warmup_hard_epochs else f"E={n_easy}/H={n_hard}"
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f} [{phase_str}]")

        # =============================================================
        # Eval
        # =============================================================
        with torch.no_grad():
            for key, (x_test, y_test) in eval_datasets.items():
                x_test = x_test.float().to(device)
                pred = model(x_test).detach().cpu()
                if pred.ndim > 2:
                    if not warning_printed:
                        print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                        warning_printed = True
                    pred = pred.mean(1)
                if len(pred.shape) > 1:
                    pred = torch.argmax(pred, dim=-1)
                if len(y_test.shape) > 1:
                    y_test = torch.argmax(y_test, dim=-1)
                y_test = y_test.to(pred.device)
                acc = (pred == y_test).float().mean()
                results[f"{key}_all"].append(acc)

        ### print w2sg vs weak agreement/disagreement every 5 epochs
        if (epoch + 1) % 5 == 0 and "test" in eval_datasets and "test_weak" in eval_datasets:
            for split_name, gt_key, weak_key, weak_raw_key in [
                ("Train w2sg", "val", "val_weak", "val_weak_raw"),
                ("Test", "test", "test_weak", "test_weak_raw"),
            ]:
                with torch.no_grad():
                    x_t, y_t = eval_datasets[gt_key]
                    _, yw_t = eval_datasets[weak_key]
                    weak_raw_available = weak_raw_key in eval_datasets
                    if weak_raw_available:
                        _, yw_t_raw = eval_datasets[weak_raw_key]
                        if type(yw_t_raw) == np.ndarray:
                            yw_t_raw = torch.tensor(yw_t_raw, device=device)
                        elif type(yw_t_raw) == torch.Tensor:
                            yw_t_raw = yw_t_raw.to(device)

                    x_t = x_t.float().to(device)
                    logits_t = model(x_t).detach().cpu()
                    if logits_t.ndim > 2:
                        logits_t = logits_t.mean(1)

                    probs_t = torch.softmax(logits_t, dim=-1)
                    log_probs_t = torch.log(probs_t + 1e-8)
                    entropy_t = -(probs_t * log_probs_t).sum(dim=-1)
                    pred_t = torch.argmax(logits_t, dim=-1)

                    if weak_raw_available:
                        if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                            weak_probs_t = yw_t_raw
                        else:
                            weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                        weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                    else:
                        weak_confidence_t = torch.zeros(len(y_t))

                    y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                    yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t
                    y_t_flat = y_t_flat.cpu()
                    yw_t_flat = yw_t_flat.cpu()
                    w2sg_correct = (pred_t == y_t_flat)
                    weak_correct = (yw_t_flat == y_t_flat)
                    n_t = len(y_t_flat)

                    mask_both_correct = w2sg_correct & weak_correct
                    mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                    mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                    mask_both_wrong = ~w2sg_correct & ~weak_correct

                    both_correct = mask_both_correct.float().sum().item() / n_t * 100
                    w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_t * 100
                    w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_t * 100
                    both_wrong = mask_both_wrong.float().sum().item() / n_t * 100

                    ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                    ent_w2sg_rw = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                    ent_w2sg_wr = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                    ent_bw = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                    confidence_t = probs_t.max(dim=-1).values
                    conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                    conf_w2sg_rw = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                    conf_w2sg_wr = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                    conf_bw = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                    weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                    weak_conf_w2sg_rw = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                    weak_conf_w2sg_wr = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                    weak_conf_bw = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                    print(f"  [Epoch {epoch+1}] {split_name} set breakdown:")
                    print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                    print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_rw:.4f}  | w2sg conf: {conf_w2sg_rw:.4f} | weak conf: {weak_conf_w2sg_rw:.4f}")
                    print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wr:.4f}  | w2sg conf: {conf_w2sg_wr:.4f} | weak conf: {weak_conf_w2sg_wr:.4f}")
                    print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_bw:.4f}  | w2sg conf: {conf_bw:.4f} | weak conf: {weak_conf_bw:.4f}")

    ### final results
    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]

    results['_balance_before'] = balance_before
    results['_balance_after'] = balance_after

    return results, model


# =============================================================================
# train_logreg_entropy_hard: Curriculum learning with confidence + entropy split
# =============================================================================
def train_logreg_entropy_hard(
    x_train,
    y_train,
    eval_datasets,
    device,
    loss_fn,
    n_epochs=20,
    weight_decay=0.0,
    lr=1e-3,
    batch_size=128,
    n_classes=1000,
    sample_weights=None,
    before_batch_callback=None,
    after_batch_callback=None,
    # --- Hard sample curriculum params ---
    warmup_hard_epochs=5,
    cluster_method='gmm',        # 'gmm', 'kmeans', 'threshold', 'median'
    weak_conf_threshold=0.5,
    w2sg_conf_threshold=0.5,
    # --- Entropy-based D2 params ---
    entropy_threshold=1.5,
    d0_weight=1.0,               # weight for D0 (high strong-conf, easy)
    d1_weight=2.0,               # weight for D1 (low strong-conf, hard)
    d2_weight=3.0,               # weight for D2 (high entropy, most uncertain)
):
    """
    train_logreg with hard sample + entropy curriculum learning:
      Phase 0: Filter data by weak model confidence clustering (same as train_logreg_hard)
      Epochs 0 ~ warmup_hard_epochs-1: Train on filtered data with equal weights
      From warmup_hard_epochs:
        - D0 = filtered data where w2sg has HIGH confidence (easy)
        - D1 = filtered data where w2sg has LOW confidence (hard)
        - D2 = samples from ORIGINAL unfiltered training data where w2sg entropy >= entropy_threshold
        - Final training set = union(D0, D1, D2), with weights d0_weight < d1_weight < d2_weight
        - Overlapping samples (in both filtered set and D2) take max weight (d2_weight)
    """

    # Keep a reference to the ORIGINAL unfiltered data for D2 extraction later
    x_train_original = x_train.clone() if isinstance(x_train, torch.Tensor) else x_train.copy()
    y_train_original = y_train.clone() if isinstance(y_train, torch.Tensor) else y_train.copy()
    n_original = len(x_train_original)

    # =================================================================
    # PHASE 0: Compute weak confidence and filter training data
    # =================================================================
    print(f"\n{'='*60}")
    print(f"[Entropy Hard Curriculum] Phase 0: Filtering by weak confidence")
    print(f"  Cluster method: {cluster_method}, threshold: {weak_conf_threshold}")
    print(f"  Entropy threshold for D2: {entropy_threshold}")
    print(f"  Weights: D0={d0_weight}, D1={d1_weight}, D2={d2_weight}")
    print(f"{'='*60}")

    # Compute weak confidence from y_train (= weak labels)
    if y_train.ndim == 3:
        yw_soft = y_train.float().mean(1)  # (N, K, C) -> (N, C)
    else:
        yw_soft = y_train.float()  # (N, C)

    # Check if probabilities or logits
    if yw_soft.min() >= 0 and yw_soft.max() <= 1.01:
        weak_probs = yw_soft
    else:
        weak_probs = torch.softmax(yw_soft, dim=-1)

    weak_confidence = weak_probs.max(dim=-1).values.cpu().numpy()

    # Cluster and filter (v2: ensures min samples per class)
    keep_mask, weak_cluster_info = cluster_by_confidence_v2(
        weak_confidence, labels=y_train, n_classes=n_classes,
        method=cluster_method, threshold=weak_conf_threshold,
        min_samples_per_class=5
    )

    n_before = len(x_train)
    x_train_filtered = x_train[keep_mask]
    y_train_filtered = y_train[keep_mask]
    # Track which original indices are in the filtered set
    filtered_original_indices = np.where(keep_mask)[0]
    n_after = len(x_train_filtered)

    # --- Class balance stats BEFORE filtering ---
    balance_before = compute_class_balance_stats(y_train, n_classes)
    print(f"  === Class Balance BEFORE filtering ===")
    print(format_balance_stats(balance_before, prefix="    "))

    print(f"  Samples before filtering: {n_before}")
    print(f"  Samples after filtering:  {n_after} ({n_after/n_before*100:.1f}%)")
    print(f"  Removed: {n_before - n_after} samples ({(n_before - n_after)/n_before*100:.1f}%)")
    print(f"  Weak confidence stats: min={weak_confidence.min():.4f}, max={weak_confidence.max():.4f}, mean={weak_confidence.mean():.4f}")
    print(f"  Cluster info: {weak_cluster_info}")

    # --- Class balance stats AFTER filtering ---
    balance_after = compute_class_balance_stats(y_train_filtered, n_classes)
    print(f"  === Class Balance AFTER filtering ===")
    print(format_balance_stats(balance_after, prefix="    "))

    # =================================================================
    # Setup: start with filtered data, model, optimizer
    # =================================================================
    x_train_filtered = x_train_filtered.float()
    x_train_original = x_train_original.float()

    # Initial dataset uses filtered data only (warmup phase)
    weights_tensor = torch.ones(n_after, device=device)
    train_ds = torch.utils.data.TensorDataset(
        x_train_filtered,
        y_train_filtered,
        torch.arange(n_after),
        weights_tensor,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    ### setup model and optimizer
    model = torch.nn.Linear(x_train_filtered.shape[-1], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    iter_i = 0
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)
    warning_printed = False

    ### train and eval
    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    results["entropy_hard_curriculum_info"] = []

    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):

        # =============================================================
        # PHASE 1 & 2: Update data and weights based on epoch
        # =============================================================
        if epoch >= warmup_hard_epochs:
            # ---- PHASE 2: Confidence + Entropy curriculum ----
            model.eval()

            # --- Step 1: Cluster filtered data into D0 (easy) and D1 (hard) by w2sg confidence ---
            with torch.no_grad():
                logits_filtered = model(x_train_filtered.to(device))
                w2sg_probs_filtered = torch.softmax(logits_filtered, dim=-1)
                w2sg_confidence_filtered = w2sg_probs_filtered.max(dim=-1).values.cpu().numpy()

            easy_mask, w2sg_cluster_info = cluster_by_confidence(
                w2sg_confidence_filtered, method=cluster_method, threshold=w2sg_conf_threshold
            )
            # D0 = easy (high confidence), D1 = hard (low confidence) within filtered data
            d0_mask_in_filtered = easy_mask        # boolean array over filtered data
            d1_mask_in_filtered = ~easy_mask

            # --- Step 2: Find D2 from ORIGINAL data based on w2sg entropy ---
            with torch.no_grad():
                logits_original = model(x_train_original.to(device))
                w2sg_probs_original = torch.softmax(logits_original, dim=-1)
                log_probs_original = torch.log(w2sg_probs_original + 1e-8)
                entropy_original = -(w2sg_probs_original * log_probs_original).sum(dim=-1).cpu().numpy()

            # D2: samples from original data with entropy >= threshold
            d2_mask_in_original = entropy_original >= entropy_threshold
            d2_original_indices = np.where(d2_mask_in_original)[0]

            # --- Step 3: Build the union dataset D0 ∪ D1 ∪ D2 ---
            # Start with filtered set (D0 ∪ D1), then add D2 samples not already in filtered set
            filtered_set = set(filtered_original_indices.tolist())
            d2_new_indices = [idx for idx in d2_original_indices if idx not in filtered_set]

            # Build combined data
            if len(d2_new_indices) > 0:
                x_d2_new = x_train_original[d2_new_indices]
                y_d2_new = y_train_original[d2_new_indices]
                x_combined = torch.cat([x_train_filtered, x_d2_new.float()], dim=0)
                y_combined = torch.cat([y_train_filtered, y_d2_new], dim=0)
            else:
                x_combined = x_train_filtered
                y_combined = y_train_filtered

            n_combined = len(x_combined)

            # Build weight vector for combined data
            combined_weights = torch.ones(n_combined, device=device)

            # Assign weights for filtered portion: D0 -> d0_weight, D1 -> d1_weight
            for i in range(n_after):
                if d0_mask_in_filtered[i]:
                    combined_weights[i] = d0_weight
                else:
                    combined_weights[i] = d1_weight

            # For D2 samples that are ALSO in filtered set, upgrade weight to d2_weight
            d2_overlap_set = set(d2_original_indices.tolist()) & filtered_set
            if d2_overlap_set:
                # Map original indices to filtered indices
                orig_to_filtered = {orig_idx: filt_idx for filt_idx, orig_idx in enumerate(filtered_original_indices)}
                for orig_idx in d2_overlap_set:
                    filt_idx = orig_to_filtered[orig_idx]
                    combined_weights[filt_idx] = d2_weight  # upgrade to max

            # For newly added D2 samples (not in filtered set)
            if len(d2_new_indices) > 0:
                combined_weights[n_after:] = d2_weight

            # Rebuild dataloader with combined data
            train_ds_epoch = torch.utils.data.TensorDataset(
                x_combined,
                y_combined,
                torch.arange(n_combined),
                combined_weights,
            )
            train_loader = torch.utils.data.DataLoader(train_ds_epoch, shuffle=True, batch_size=batch_size)
            n_batches = len(train_loader)

            n_d0 = int(d0_mask_in_filtered.sum())
            n_d1 = int(d1_mask_in_filtered.sum())
            n_d2_overlap = len(d2_overlap_set)
            n_d2_new = len(d2_new_indices)
            n_d2_total = int(d2_mask_in_original.sum())

            curriculum_info = {
                'epoch': epoch,
                'phase': 'entropy_hard_curriculum',
                'n_d0': n_d0,
                'n_d1': n_d1,
                'n_d2_total': n_d2_total,
                'n_d2_overlap': n_d2_overlap,
                'n_d2_new': n_d2_new,
                'n_combined': n_combined,
                'w2sg_cluster_info': w2sg_cluster_info,
                'avg_entropy_original': float(entropy_original.mean()),
                'entropy_threshold': entropy_threshold,
            }
            results["entropy_hard_curriculum_info"].append(curriculum_info)

            if epoch == warmup_hard_epochs or (epoch + 1) % 5 == 0:
                print(f"\n  [Epoch {epoch}] Entropy Hard Curriculum ({cluster_method}):")
                print(f"    D0 (easy, w={d0_weight}): {n_d0} samples")
                print(f"    D1 (hard, w={d1_weight}): {n_d1} samples")
                print(f"    D2 (high entropy >= {entropy_threshold}, w={d2_weight}): {n_d2_total} total "
                      f"({n_d2_overlap} overlap with filtered, {n_d2_new} newly added)")
                print(f"    Combined training set: {n_combined} samples")
                print(f"    Avg entropy (all original): {entropy_original.mean():.4f}")
                print(f"    W2SG cluster info: {w2sg_cluster_info}")
        else:
            # ---- PHASE 1: Warmup with equal weights ----
            # Use filtered data with equal weights, dataloader stays the same
            weights_tensor.fill_(1.0)
            results["entropy_hard_curriculum_info"].append({
                'epoch': epoch, 'phase': 'warmup', 'n_samples': n_after
            })

        # =============================================================
        # Training loop
        # =============================================================
        model.train()
        correct, total = 0, 0
        for b_i, (x, y, sample_idxs, sample_ws) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            if before_batch_callback is not None:
                x, y, pred, sample_ws = before_batch_callback(x=x, y=y, pred=pred, sample_idxs=sample_idxs, sample_ws=sample_ws, epoch=epoch, is_eval=False)
            if pred.ndim > 2:
                if not warning_printed:
                    print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                    warning_printed = True
                pred = pred.mean(1)
            loss = loss_fn(pred, y, step_frac=(iter_i := iter_i + 1) / n_iter, sample_weights=sample_ws)
            loss.backward()
            optimizer.step()
            schedule.step()

            ### calc metrics
            if len(y.shape) == 2:
                y = torch.argmax(y, dim=-1)
            elif len(y.shape) == 3:
                y = y.mean(1).argmax(-1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
            if after_batch_callback is not None:
                after_batch_callback(x=x, y=y, pred=pred, loss=loss, last_in_epoch=b_i == n_batches - 1, epoch=epoch)

        if epoch < warmup_hard_epochs:
            phase_str = "warmup"
        else:
            phase_str = f"D0={n_d0}/D1={n_d1}/D2={n_d2_total}"
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f} [{phase_str}]")

        # =============================================================
        # Eval
        # =============================================================
        with torch.no_grad():
            for key, (x_test, y_test) in eval_datasets.items():
                x_test = x_test.float().to(device)
                pred = model(x_test).detach().cpu()
                if pred.ndim > 2:
                    if not warning_printed:
                        print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                        warning_printed = True
                    pred = pred.mean(1)
                if len(pred.shape) > 1:
                    pred = torch.argmax(pred, dim=-1)
                if len(y_test.shape) > 1:
                    y_test = torch.argmax(y_test, dim=-1)
                # Ensure device match
                y_test = y_test.to(pred.device)
                acc = (pred == y_test).float().mean()
                results[f"{key}_all"].append(acc)

        ### print w2sg vs weak agreement/disagreement on test set every 5 epochs
        if (epoch + 1) % 5 == 0 and "test" in eval_datasets and "test_weak" in eval_datasets:
            with torch.no_grad():
                x_t, y_t = eval_datasets["val"]
                _, yw_t = eval_datasets["val_weak"]
                weak_raw_available = "val_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["val_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = model(x_t).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)

                pred_t = torch.argmax(logits_t, dim=-1)

                if weak_raw_available:
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t

                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                confidence_t = probs_t.max(dim=-1).values
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Train w2sg set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")

            with torch.no_grad():
                x_t, y_t = eval_datasets["test"]
                _, yw_t = eval_datasets["test_weak"]
                weak_raw_available = "test_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["test_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = model(x_t).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)

                pred_t = torch.argmax(logits_t, dim=-1)

                if weak_raw_available:
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t

                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                confidence_t = probs_t.max(dim=-1).values
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Test set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")

    ### final results
    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]

    # Store balance stats for visualization
    results['_balance_before'] = balance_before
    results['_balance_after'] = balance_after

    return results, model


from rw2s._disme_funcs import CPPProjector, supervised_contrastive_loss, topology_regularization_loss, NearestCentroidClassifier

# =============================================================================
# train_logreg_hard_disme: CPP-STPR with hard sample curriculum
# =============================================================================
def train_logreg_hard_disme(
    x_train, y_train, eval_datasets, device, loss_fn,
    n_epochs=20, weight_decay=0.0, lr=1e-3, batch_size=128, n_classes=1000,
    sample_weights=None, before_batch_callback=None, after_batch_callback=None,
    warmup_hard_epochs=3, hard_weight=2.0, cluster_method='gmm',
    weak_conf_threshold=0.5, w2sg_conf_threshold=0.5,
    projector_hidden_dim=512, projector_output_dim=256,
    supcon_temperature=0.07, stpr_temperature=1.0,
    lambda_supcon=1.0, lambda_stpr=0.5,
):
    """
    CPP-STPR with hard sample curriculum learning.
    Phase 0: Filter data by weak confidence (cluster_by_confidence_v2).
    Warmup: Train projector with equal weights.
    After warmup: Split easy/hard by NCC confidence, upweight hard samples.
    Uses Projector + SupCon + STPR + Nearest Centroid Classifier.
    """
    import torch.nn.functional as F

    print(f"\n{'='*60}")
    print(f"[CPP-STPR Hard Curriculum] Phase 0: Filtering by weak confidence")
    print(f"  Cluster method: {cluster_method}, threshold: {weak_conf_threshold}")
    print(f"  Projector: {x_train.shape[-1]} -> {projector_hidden_dim} -> {projector_output_dim}")
    print(f"  SupCon temp: {supcon_temperature}, STPR temp: {stpr_temperature}")
    print(f"  Lambda SupCon: {lambda_supcon}, Lambda STPR: {lambda_stpr}")
    print(f"{'='*60}")

    yw_soft = y_train.float().mean(1) if y_train.ndim == 3 else y_train.float()
    weak_probs = yw_soft if (yw_soft.min() >= 0 and yw_soft.max() <= 1.01) else torch.softmax(yw_soft, dim=-1)
    weak_confidence = weak_probs.max(dim=-1).values.cpu().numpy()

    keep_mask, weak_cluster_info = cluster_by_confidence_v2(
        weak_confidence, labels=y_train, n_classes=n_classes,
        method=cluster_method, threshold=weak_conf_threshold, min_samples_per_class=5
    )

    n_before = len(x_train)
    x_filt = x_train[keep_mask]
    y_filt = y_train[keep_mask]
    n_after = len(x_filt)

    balance_before = compute_class_balance_stats(y_train, n_classes)
    print(f"  === Class Balance BEFORE filtering ===")
    print(format_balance_stats(balance_before, prefix="    "))
    print(f"  Samples: {n_before} -> {n_after} ({n_after/n_before*100:.1f}%), removed {n_before-n_after}")
    balance_after = compute_class_balance_stats(y_filt, n_classes)
    print(f"  === Class Balance AFTER filtering ===")
    print(format_balance_stats(balance_after, prefix="    "))

    x_filt = x_filt.float()
    if y_filt.ndim == 3:
        y_hard = y_filt.float().mean(1).argmax(-1)
    elif y_filt.ndim == 2:
        y_hard = y_filt.argmax(-1)
    else:
        y_hard = y_filt

    input_dim = x_filt.shape[-1]
    projector = CPPProjector(input_dim, projector_hidden_dim, projector_output_dim).to(device)
    ncc = NearestCentroidClassifier(n_classes)
    weights_tensor = torch.ones(n_after)

    train_ds = torch.utils.data.TensorDataset(x_filt, y_hard, torch.arange(n_after), weights_tensor)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    optimizer = torch.optim.Adam(projector.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    iter_i = 0
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)

    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    results["hard_curriculum_info"] = []

    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):
        if epoch >= warmup_hard_epochs:
            projector.eval()
            with torch.no_grad():
                v_all = projector(x_filt.to(device))
            ncc.fit(v_all, y_hard.to(device))
            with torch.no_grad():
                ncc_conf = torch.softmax(ncc.predict_with_logits(v_all), dim=-1).max(-1).values.cpu().numpy()
            easy_mask, _ = cluster_by_confidence(ncc_conf, method=cluster_method, threshold=w2sg_conf_threshold)
            hard_indices = torch.from_numpy(~easy_mask).bool()
            weights_tensor.fill_(1.0)
            weights_tensor[hard_indices] = hard_weight
            n_easy, n_hard = int(easy_mask.sum()), int((~easy_mask).sum())
            results["hard_curriculum_info"].append({'epoch': epoch, 'phase': 'hard_curriculum', 'n_easy': n_easy, 'n_hard': n_hard})
            if epoch == warmup_hard_epochs or (epoch + 1) % 5 == 0:
                print(f"\n  [Epoch {epoch}] CPP-STPR Split: Easy={n_easy} Hard={n_hard} (weight={hard_weight})")
        else:
            weights_tensor.fill_(1.0)
            n_easy, n_hard = n_after, 0
            results["hard_curriculum_info"].append({'epoch': epoch, 'phase': 'warmup', 'n_samples': n_after})

        projector.train()
        ep_supcon, ep_stpr = 0.0, 0.0
        for b_i, (x, y, sidx, sw) in enumerate(train_loader):
            x, y, sw = x.to(device), y.to(device), sw.to(device)
            optimizer.zero_grad()
            z = x
            v = projector(z)
            v_norm = F.normalize(v, dim=-1)
            l_sup = supervised_contrastive_loss(v_norm, y, temperature=supcon_temperature)
            l_stp = topology_regularization_loss(z, v, y, n_classes, tau=stpr_temperature)
            w_mean = sw.mean()
            loss = lambda_supcon * l_sup * w_mean + lambda_stpr * l_stp
            loss.backward()
            optimizer.step()
            schedule.step()
            iter_i += 1
            ep_supcon += l_sup.item()
            ep_stpr += l_stp.item()

        projector.eval()
        with torch.no_grad():
            v_all = projector(x_filt.to(device))
        ncc.fit(v_all, y_hard.to(device))
        with torch.no_grad():
            train_acc = (ncc.predict(v_all) == y_hard.to(device)).float().mean().item()

        phase_str = "warmup" if epoch < warmup_hard_epochs else f"E={n_easy}/H={n_hard}"
        pbar.set_description(f"Epoch {epoch}, Acc {train_acc:.3f}, SC {ep_supcon/max(n_batches,1):.4f}, ST {ep_stpr/max(n_batches,1):.4f} [{phase_str}]")

        with torch.no_grad():
            for key, (xt, yt) in eval_datasets.items():
                vt = projector(xt.float().to(device))
                pred = ncc.predict(vt).detach().cpu()
                yt_f = yt.argmax(-1) if yt.ndim > 1 else yt
                results[f"{key}_all"].append((pred == yt_f.to(pred.device)).float().mean())

        if (epoch + 1) % 5 == 0 and "test" in eval_datasets and "test_weak" in eval_datasets:
            for sn, gk, wk, wrk in [("Val", "val", "val_weak", "val_weak_raw"), ("Test", "test", "test_weak", "test_weak_raw")]:
                with torch.no_grad():
                    xt2, yt2 = eval_datasets[gk]
                    _, ywt = eval_datasets[wk]
                    lt = ncc.predict_with_logits(projector(xt2.float().to(device))).detach().cpu()
                    pt2 = torch.softmax(lt, -1)
                    ent = -(pt2 * torch.log(pt2 + 1e-8)).sum(-1)
                    predt = lt.argmax(-1)
                    conf = pt2.max(-1).values
                    ytf = yt2.argmax(-1).cpu() if yt2.ndim > 1 else yt2.cpu()
                    ywf = ywt.argmax(-1).cpu() if ywt.ndim > 1 else ywt.cpu()
                    sc = (predt == ytf); wc = (ywf == ytf); nt = len(ytf)
                    mbc = sc & wc; mwc = sc & ~wc; mcw = ~sc & wc; mbw = ~sc & ~wc
                    def _p(m): return m.float().sum().item()/nt*100
                    def _a(v2, m): return v2[m].mean().item() if m.any() else float('nan')
                    print(f"  [Epoch {epoch+1}] {sn}: BC={_p(mbc):.1f}% WC={_p(mwc):.1f}% CW={_p(mcw):.1f}% BW={_p(mbw):.1f}%")
            with torch.no_grad():
                x_t, y_t = eval_datasets["val"]
                _, yw_t = eval_datasets["val_weak"]  # yw_t are weak hard labels
                weak_raw_available = "val_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["val_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = ncc.predict_with_logits(projector(x_t)).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                # Compute softmax probabilities and entropy: H = -sum(p * log(p))
                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)  # add eps to avoid log(0)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)  # shape: (n_test,)

                pred_t = torch.argmax(logits_t, dim=-1)
                
                # We need to compute weak confidence
                if weak_raw_available:
                    # yw_t_raw might be logits or probabilities
                    # To be safe, if min >= 0 and max <= 1.01 it's likely probs, else logits
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t
                
                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                # Masks for 4 cases
                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                # Average entropy for each case
                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                # Average confidence (max softmax probability) for each case
                confidence_t = probs_t.max(dim=-1).values  # shape: (n_test,)
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')
                
                # Average weak confidence for each case
                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Train w2sg set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")

            with torch.no_grad():
                x_t, y_t = eval_datasets["test"]
                _, yw_t = eval_datasets["test_weak"]  # yw_t are weak hard labels
                weak_raw_available = "test_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["test_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = ncc.predict_with_logits(projector(x_t)).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                # Compute softmax probabilities and entropy
                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)

                pred_t = torch.argmax(logits_t, dim=-1)

                # Compute weak confidence
                if weak_raw_available:
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t

                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                # Masks for 4 cases
                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                # Average entropy for each case
                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                # Average confidence (max softmax probability) for each case
                confidence_t = probs_t.max(dim=-1).values
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                # Average weak confidence for each case
                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Test set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")


    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]
    results['_balance_before'] = balance_before
    results['_balance_after'] = balance_after
    return results, {'projector': projector, 'ncc': ncc, 'type': 'cpp_stpr'}


# =============================================================================
# train_head_DG_hard_disme: DG wrapper calling train_logreg_hard_disme
# =============================================================================
def train_head_DG_hard_disme(
    teacher_model, student_model, val_dataloader, test_dataloader,
    cfg, logger, cached_labels_path, cached_embs_path, results, rng, n_classes,
    return_data=False, additional_eval_data=None,
    before_optim_run_callback_weak=None, before_optim_run_callback_gt=None,
    after_batch_callback_weak=None, before_batch_callback_weak=None,
    after_batch_callback_gt=None, before_batch_callback_gt=None,
):
    """
    train_head_DG with CPP-STPR hard sample curriculum learning.
    Same data loading/splitting as train_head_DG_hard but calls
    train_logreg_hard_disme (Projector + SupCon + STPR + NCC) for w2s training.
    """
    ### get (weak) labels from current teacher
    if cfg["w2s"]["load_labels"] and os.path.exists(cached_labels_path):
        logger.info("Loading teacher labels from cache...")
        cached = torch.load(cached_labels_path, pickle_module=dill, map_location="cpu")
        val_gt_labels, val_teacher_labels = cached["val_gt_labels"], cached["val_teacher_labels"]
        test_gt_labels, test_teacher_labels = cached["test_gt_labels"], cached["test_teacher_labels"]
        teacher_acc = cached.get("teacher_acc", 0)
    else:
        chunking_dir = os.path.join(os.path.dirname(cached_labels_path), f"label_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info("Collecting teacher labels for Validation Set...")
        _, val_gt_labels, val_teacher_labels, val_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=val_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'val'))
        logger.info("Collecting teacher labels for Test Set...")
        _, test_gt_labels, test_teacher_labels, test_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=test_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'test'))
        teacher_acc = float((np.mean(val_teacher_acc) + np.mean(test_teacher_acc)) / 2.0)
        if cfg["w2s"]["save_labels"]:
            torch.save({"cfg": cfg, "val_gt_labels": val_gt_labels, "val_teacher_labels": val_teacher_labels, "test_gt_labels": test_gt_labels, "test_teacher_labels": test_teacher_labels, "teacher_acc": teacher_acc}, cached_labels_path, pickle_module=dill)

    ### get embeddings from the student model
    if cfg["w2s"]["load_embeddings"] and os.path.exists(cached_embs_path):
        logger.info("Loading student model embeddings from cache...")
        cached = torch.load(cached_embs_path, pickle_module=dill)
        val_student_embeddings, val_student_gt_labels = cached["val_embeddings"], cached["val_gt_labels"]
        test_student_embeddings, test_student_gt_labels = cached["test_embeddings"], cached["test_gt_labels"]
    else:
        chunking_dir = os.path.join(os.path.dirname(cached_embs_path), f"embs_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info("Collecting student embeddings for Validation Set...")
        val_student_embeddings, val_student_gt_labels, _, _, _, _ = preload(model=student_model, loader=val_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'val'), store_embs=True)
        logger.info("Collecting student embeddings for Test Set...")
        test_student_embeddings, test_student_gt_labels, _, _, _, _ = preload(model=student_model, loader=test_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'test'), store_embs=True)
        if cfg["w2s"]["save_embeddings"]:
            torch.save({"cfg": cfg, "val_embeddings": val_student_embeddings, "val_gt_labels": val_student_gt_labels, "test_embeddings": test_student_embeddings, "test_gt_labels": test_student_gt_labels}, cached_embs_path, pickle_module=dill)

    assert torch.all(val_gt_labels == val_student_gt_labels), "Val GT labels mismatch."
    assert torch.all(test_gt_labels == test_student_gt_labels), "Test GT labels mismatch."
    del val_student_gt_labels, test_student_gt_labels

    ### Shuffle and split VAL data
    order = np.arange(len(val_gt_labels))
    rng.shuffle(order)
    results["order"].append(order)
    x_val_all = val_student_embeddings[order]
    y_val_all = val_gt_labels[order]
    yw_val_all = val_teacher_labels[order]

    assert len(cfg["w2s"]["train_val_test_split_DG"]) == 2
    assert sum(cfg["w2s"]["train_val_test_split_DG"]) == 1.0
    n_train = int(cfg["w2s"]["train_val_test_split_DG"][0] * len(x_val_all))

    x_train, x_val = x_val_all[:n_train], x_val_all[n_train:]
    y_train, y_val = y_val_all[:n_train], y_val_all[n_train:]
    yw_train, yw_val = yw_val_all[:n_train], yw_val_all[n_train:]
    x_test, y_test, yw_test = test_student_embeddings, test_gt_labels, test_teacher_labels

    x = torch.cat([x_train, x_val, x_test]) if isinstance(x_train, torch.Tensor) else np.concatenate([x_train, x_val, x_test])
    y = torch.cat([y_train, y_val, y_test]) if isinstance(y_train, torch.Tensor) else np.concatenate([y_train, y_val, y_test])
    yw = torch.cat([yw_train, yw_val, yw_test]) if isinstance(yw_train, torch.Tensor) else np.concatenate([yw_train, yw_val, yw_test])

    yw_val_raw = yw_val.mean(1) if yw_val.ndim == 3 else yw_val
    yw_val = yw_val_raw.argmax(-1)
    yw_test_raw = yw_test.mean(1) if yw_test.ndim == 3 else yw_test
    yw_test = yw_test_raw.argmax(-1)

    eval_datasets = {"val": (x_val, y_val), "val_weak": (x_val, yw_val), "val_weak_raw": (x_val, yw_val_raw), "test": (x_test, y_test), "test_weak": (x_test, yw_test), "test_weak_raw": (x_test, yw_test_raw)}
    if additional_eval_data is not None:
        for k, v in additional_eval_data.items():
            eval_datasets[k] = v

    logger.info(f"\nTotal: {len(x)}. Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    ### eval teacher
    results["teacher_acc_src"].append(teacher_acc)
    teacher_acc_all = (y == (yw if yw.ndim == 2 else yw.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc"].append(teacher_acc_all)
    results["teacher_acc_train"].append((y_train == (yw_train if yw_train.ndim == 2 else yw_train.mean(1)).argmax(-1)).float().mean())
    results["teacher_acc_val"].append((y_val == yw_val).float().mean())
    results["teacher_acc_test"].append((y_test == yw_test).float().mean())
    if type(teacher_acc) == float:
        teacher_acc = torch.tensor([teacher_acc], device=cfg["device"])
    logger.info(f"Teacher acc - all: {teacher_acc_all:.4f}")

    ### w2s with CPP-STPR hard curriculum
    if before_optim_run_callback_weak is not None:
        before_optim_run_callback_weak(yw=yw_train, sample_idxs=np.arange(len(yw_train)))
    seed_all(cfg["seed"])

    hard_kwargs = {
        'warmup_hard_epochs': cfg["w2s"].get("warmup_hard_epochs", 5),
        'hard_weight': cfg["w2s"].get("hard_weight", 2.0),
        'cluster_method': cfg["w2s"].get("cluster_method", "kmeans"),
        'weak_conf_threshold': cfg["w2s"].get("weak_conf_threshold", 0.6),
        'w2sg_conf_threshold': cfg["w2s"].get("w2sg_conf_threshold", 0.6),
    }
    disme_kwargs = {
        'projector_hidden_dim': cfg["w2s"].get("projector_hidden_dim", 512),
        'projector_output_dim': cfg["w2s"].get("projector_output_dim", 256),
        'supcon_temperature': cfg["w2s"].get("supcon_temperature", 0.07),
        'stpr_temperature': cfg["w2s"].get("stpr_temperature", 1.0),
        'lambda_supcon': cfg["w2s"].get("lambda_supcon", 1.0),
        'lambda_stpr': cfg["w2s"].get("lambda_stpr", 0.5),
    }
    logger.info(f"CPP-STPR params: {hard_kwargs} | {disme_kwargs}")

    results_teacher_to_student, student_model_probe = train_logreg_hard_disme(
        x_train, yw_train, eval_datasets, device=cfg["device"],
        batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["teacher_labels_loss_fn_name"]](**(cfg["w2s"]["teacher_labels_loss_fn_kwargs"] or dict())),
        n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None,
        before_batch_callback=before_batch_callback_weak,
        after_batch_callback=after_batch_callback_weak,
        **hard_kwargs, **disme_kwargs,
    )
    results["results_teacher_to_student"].append(results_teacher_to_student)
    results["student_model_probe"].append(student_model_probe)

    ### gt
    if before_optim_run_callback_gt is not None:
        before_optim_run_callback_gt(yw=yw_test, sample_idxs=len(y_train) + len(y_val) + np.arange(len(yw_test)))
    seed_all(cfg["seed"])
    results_gt, gt_model_probe = train_logreg(x_train, y_train, eval_datasets, device=cfg["device"], batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["gt_labels_loss_fn_name"]](**(cfg["w2s"]["gt_labels_loss_fn_kwargs"] or dict())), n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None, before_batch_callback=before_batch_callback_gt, after_batch_callback=after_batch_callback_gt)
    results["results_gt"].append(results_gt)

    ### W2SG visualization
    _balance_stats = None
    if '_balance_before' in results_teacher_to_student and '_balance_after' in results_teacher_to_student:
        _balance_stats = {'before': results_teacher_to_student['_balance_before'], 'after': results_teacher_to_student['_balance_after']}
    if cfg["w2s"].get("plot_w2sg", False):
        _plot_dir = cfg["w2s"].get("plot_save_dir", None)
        if _plot_dir:
            _dim_methods = cfg["w2s"].get("plot_dim_reduction", ["pca"])
            if isinstance(_dim_methods, str):
                _dim_methods = [_dim_methods]
            for _dm in _dim_methods:
                try:
                    plot_w2sg_analysis(model=student_model_probe, gt_model=gt_model_probe, eval_datasets=eval_datasets, save_dir=_plot_dir, seed=cfg["seed"], device=cfg["device"], dim_reduction=_dm, balance_stats=_balance_stats, logger=logger)
                except Exception as _plot_err:
                    logger.info(f"[W2SG Plot] Warning: plotting failed ({_dm}): {_plot_err}")

    if return_data:
        return results, student_model_probe, {"x": x, "y": y, "yw": yw, "x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val, "x_test": x_test, "y_test": y_test, "yw_train": yw_train, "yw_val": yw_val, "yw_test": yw_test}
    return results, student_model_probe


def train(
    model,
    train_dl,
    loss_fn,
    val_dl=None,
    val_loss_fn=None,
    weights=None, # torch.Tensor of shape (N,)
    normalize_weights_per_batch=False,
    n_epochs=30,
    optimizer_name="Adam",
    optimizer_kwargs=dict(),
    scheduler_name=None,
    scheduler_mul_factor=None,
    early_stopping_patience=None,
    load_best_model=True,
    logger=None,
    wdb_run=None,
    ckpt_path=None,
    device=0,
):
    assert (early_stopping_patience is None and not load_best_model) or val_dl is not None, \
        "Validation dataloader is required for early stopping."
    assert weights is None or (weights.ndim == 1 and len(weights) == len(train_dl.dataset)), \
        "Weights must be a 1D tensor of length equal to the number of training samples."

    ### setup optimization and tracking
    opter = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_kwargs)
    n_iter = (n_batches := len(train_dl)) * n_epochs
    iter_i = 0
    if scheduler_name == None:
        schedule = None
    elif scheduler_name == "cosine":
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opter, T_max=n_iter)
    elif scheduler_name == "multiplicative":
        schedule = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=opter, lr_lambda=lambda ep: scheduler_mul_factor)
    else:
        raise ValueError(f"Scheduler name {scheduler_name} not recognized.")
    val_loss_fn = deepcopy(loss_fn) if val_loss_fn is None else val_loss_fn
    best = {"val_loss": np.inf, "val_acc": 0}
    ea_worse_epochs = 0

    ### load if ckpt exists
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, pickle_module=dill)
        model.load_state_dict(ckpt["model"])
        opter.load_state_dict(ckpt["opter"])
        if scheduler_name is not None and ckpt["scheduler"] is not None:
            schedule.load_state_dict(ckpt["scheduler"])
        epoch = ckpt["epoch"]
        iter_i = ckpt["iter_i"]
        best = ckpt["best"]
        logger.info(f"Loaded checkpoint from epoch {epoch}.")

    ### run
    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):
        correct, total, loss_ep = 0, 0, 0

        model.train()
        for b in train_dl:
            x, y = b[0].to(device), b[1].to(device)

            opter.zero_grad()
            pred = model(x)
            if len(pred) == 2 and type(pred) is not torch.Tensor:
                pred = pred[1]
            loss = loss_fn(pred, y, step_frac=(iter_i := iter_i + 1) / n_iter, reduction="none")

            ### weight loss
            if weights is not None:
                batch_weights = weights[total:total+len(y)]
                if normalize_weights_per_batch:
                    batch_weights = batch_weights / batch_weights.sum()
                loss = loss * batch_weights

            ### backprop
            loss.mean().backward()
            opter.step()
            if scheduler_name == "cosine":
                schedule.step()

            ### logging
            if len(y.shape) > 1:
                y = torch.argmax(y, dim=1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
            loss_ep += loss.sum().item()

        ### eval
        val_correct, val_total, val_loss = 0, 0, 0
        if val_dl is not None:
            model.eval()
            with torch.no_grad():
                for b in val_dl:
                    x, y = b[0].to(device), b[1].to(device)
                    pred = model(x)
                    if len(pred) == 2 and type(pred) is not torch.Tensor:
                        pred = pred[1]
                    loss = val_loss_fn(pred, y, step_frac=None, reduction="sum")

                    ### logging
                    if len(y.shape) > 1:
                        y = torch.argmax(y, dim=1)
                    val_correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
                    val_loss += loss.item()
                    val_total += len(y)

            val_loss_ep = val_loss / val_total
            val_acc = val_correct / val_total
            if wdb_run is not None:
                wdb_run.log({"val_loss": val_loss_ep, "val_acc": val_acc}, commit=False)

            ### update best
            if val_acc > best["val_acc"]:
                best["model"] = deepcopy(model.state_dict())
                best["val_loss"] = val_loss_ep
                best["val_acc"] = val_acc
                best["epoch"] = epoch
                ea_worse_epochs = 0
            else:
                ea_worse_epochs += 1

        ### logging
        if wdb_run is not None:
            wdb_run.log({"train_loss": loss_ep / total, "train_acc": correct / total})
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.4f}, Val Acc {round(val_correct / val_total, 4) if val_dl is not None else '---'}")
        logger.info(f"[{epoch}/{n_epochs}]  Train loss: {loss_ep / total:.4f}  |  Train acc: {correct / total:.4f}" +  (f"  |  Val loss: {val_loss_ep:.4f}  |  Val acc: {val_acc:.4f}" if val_dl is not None else ""))

        ### early stopping
        if early_stopping_patience is not None and ea_worse_epochs > early_stopping_patience:
            pbar.set_description(f"Early stopping. Best val loss: {best['val_loss']:.4f}. Best val acc: {best['val_acc']:.4f}.")
            logger.info(f"Early stopping. Best val loss: {best['val_loss']:.4f}. Best val acc: {best['val_acc']:.4f}.")
            break

        if scheduler_name == "multiplicative":
            schedule.step()

        ### save checkpoint
        if ckpt_path is not None:
            torch.save({
                "model": model.state_dict(),
                "opter": opter.state_dict(),
                "scheduler": schedule.state_dict() if schedule is not None else None,
                "epoch": epoch,
                "iter_i": iter_i,
                "best": best,
            }, ckpt_path, pickle_module=dill)

    ### load best model
    if load_best_model and "model" in best:
        if logger is not None:
            logger.info(f"Loading the best model from epoch {best['epoch']}. Validation loss: {best['val_loss']:.4f}. Validation accuracy: {best['val_acc']:.4f}.")
        model.load_state_dict(best["model"])

    return model


# =============================================================================
# Mixup helper: generates augmented (x_mix, y_mix) pairs from embeddings + soft labels
# =============================================================================
def mixup_data(x, y_soft, alpha=1.0, n_augment=None):
    """
    Mixup augmentation on embedding space.
    Args:
        x: (N, D) embeddings
        y_soft: (N, C) soft labels from weak model
        alpha: Beta distribution parameter (alpha=1.0 -> uniform)
        n_augment: number of augmented samples to generate (default: same as N)
    Returns: x_mix (M, D), y_mix (M, C)
    """
    N = x.shape[0]
    if n_augment is None:
        n_augment = N
    lam = torch.distributions.Beta(alpha, alpha).sample((n_augment,)).to(x.device)
    lam = lam.unsqueeze(-1)  # (M, 1)
    idx_a = torch.randint(0, N, (n_augment,))
    idx_b = torch.randint(0, N, (n_augment,))
    x_mix = lam * x[idx_a] + (1 - lam) * x[idx_b]
    y_mix = lam * y_soft[idx_a] + (1 - lam) * y_soft[idx_b]
    return x_mix, y_mix


# =============================================================================
# Mixup helper v2: same-class mixup (intra-class interpolation only)
# =============================================================================
def mixup_data_v2(x, y_soft, alpha=1.0, n_augment=None):
    """
    Same-class mixup augmentation on embedding space.
    Only interpolates between two data points that share the same predicted
    class (argmax of the weak soft label).  This keeps synthetic samples
    within the class manifold.

    Args:
        x: (N, D) embeddings
        y_soft: (N, C) soft labels from weak model (probabilities)
        alpha: Beta distribution parameter (alpha=1.0 -> uniform)
        n_augment: number of augmented samples to generate (default: same as N)
    Returns: x_mix (M, D), y_mix (M, C)   where M <= n_augment
             (may be slightly less if some classes have <2 samples)
    """
    N = x.shape[0]
    if n_augment is None:
        n_augment = N

    # --- Build per-class index ---
    weak_labels = y_soft.argmax(dim=-1)  # (N,)
    unique_classes = weak_labels.unique()
    class_to_indices = {c.item(): (weak_labels == c).nonzero(as_tuple=True)[0]
                        for c in unique_classes}

    # Only keep classes with >=2 samples (need a pair to interpolate)
    valid_classes = [c for c, idx in class_to_indices.items() if len(idx) >= 2]
    if len(valid_classes) == 0:
        # Fallback: cannot do same-class mixup, return empty tensors
        return x[:0], y_soft[:0]

    # --- Count how many samples to draw per class (proportional to class size) ---
    class_sizes = torch.tensor([len(class_to_indices[c]) for c in valid_classes],
                               dtype=torch.float)
    class_weights = class_sizes / class_sizes.sum()
    per_class_n = (class_weights * n_augment).long()
    # Distribute remainder to the largest classes
    remainder = n_augment - per_class_n.sum().item()
    if remainder > 0:
        _, top_idx = class_sizes.sort(descending=True)
        for i in range(min(remainder, len(valid_classes))):
            per_class_n[top_idx[i]] += 1

    # --- Generate intra-class mixup pairs ---
    x_mix_list = []
    y_mix_list = []
    for ci, c in enumerate(valid_classes):
        m = per_class_n[ci].item()
        if m == 0:
            continue
        indices = class_to_indices[c]
        n_c = len(indices)
        # Random pairs within the class
        local_a = torch.randint(0, n_c, (m,))
        local_b = torch.randint(0, n_c, (m,))
        idx_a = indices[local_a]
        idx_b = indices[local_b]
        lam = torch.distributions.Beta(alpha, alpha).sample((m,)).to(x.device)
        lam = lam.unsqueeze(-1)  # (m, 1)
        x_mix_list.append(lam * x[idx_a] + (1 - lam) * x[idx_b])
        y_mix_list.append(lam * y_soft[idx_a] + (1 - lam) * y_soft[idx_b])

    x_mix = torch.cat(x_mix_list, dim=0)
    y_mix = torch.cat(y_mix_list, dim=0)
    return x_mix, y_mix


# =============================================================================
# train_logreg_mixup: Confidence filtering + Mixup augmentation on embeddings
# =============================================================================
def train_logreg_mixup(
    x_train, y_train, eval_datasets, device, loss_fn,
    n_epochs=20, weight_decay=0.0, lr=1e-3, batch_size=128, n_classes=1000,
    sample_weights=None, before_batch_callback=None, after_batch_callback=None,
    # --- Mixup params ---
    cluster_method='gmm', weak_conf_threshold=0.5,
    mixup_alpha=1.0, mixup_ratio=1.0,
):
    """
    train_logreg with confidence filtering + mixup augmentation:
      1. Filter data by weak model confidence (cluster_by_confidence_v2)
      2. Generate mixup-augmented samples on embedding space with soft labels
      3. Combine filtered + augmented data for training
    """
    print(f"\n{'='*60}")
    print(f"[Mixup] Filtering by weak confidence + Mixup augmentation")
    print(f"  cluster_method={cluster_method}, threshold={weak_conf_threshold}")
    print(f"  mixup_alpha={mixup_alpha}, mixup_ratio={mixup_ratio}")
    print(f"{'='*60}")

    # --- Compute weak confidence & filter ---
    yw_soft = y_train.float().mean(1) if y_train.ndim == 3 else y_train.float()
    weak_probs = yw_soft if (yw_soft.min() >= 0 and yw_soft.max() <= 1.01) else torch.softmax(yw_soft, dim=-1)
    weak_confidence = weak_probs.max(dim=-1).values.cpu().numpy()

    keep_mask, weak_cluster_info = cluster_by_confidence_v2(
        weak_confidence, labels=y_train, n_classes=n_classes,
        method=cluster_method, threshold=weak_conf_threshold, min_samples_per_class=5
    )
    n_before = len(x_train)
    x_filtered = x_train[keep_mask].float()
    y_filtered = y_train[keep_mask]
    n_filtered = len(x_filtered)

    # Soft labels for filtered data
    y_soft_f = y_filtered.float().mean(1) if y_filtered.ndim == 3 else y_filtered.float()
    y_soft_probs = y_soft_f if (y_soft_f.min() >= 0 and y_soft_f.max() <= 1.01) else torch.softmax(y_soft_f, dim=-1)

    # --- Class balance stats ---
    balance_before = compute_class_balance_stats(y_train, n_classes)
    print(f"  === Class Balance BEFORE filtering ===")
    print(format_balance_stats(balance_before, prefix="    "))
    print(f"  Samples: {n_before} -> {n_filtered} ({n_filtered/n_before*100:.1f}%)")

    balance_after = compute_class_balance_stats(y_filtered, n_classes)
    print(f"  === Class Balance AFTER filtering ===")
    print(format_balance_stats(balance_after, prefix="    "))

    # --- Mixup augmentation ---
    n_augment = int(n_filtered * mixup_ratio)
    x_mix, y_mix = mixup_data_v2(x_filtered, y_soft_probs, alpha=mixup_alpha, n_augment=n_augment)
    x_combined = torch.cat([x_filtered, x_mix], dim=0)
    y_combined = torch.cat([y_soft_probs, y_mix], dim=0)
    n_total = len(x_combined)
    print(f"  Mixup: +{n_augment} samples, total={n_total}")

    # --- Setup ---
    weights_tensor = torch.ones(n_total, device=device)
    train_ds = torch.utils.data.TensorDataset(x_combined, y_combined, torch.arange(n_total), weights_tensor)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    model = torch.nn.Linear(x_combined.shape[-1], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    iter_i = 0
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)
    warning_printed = False

    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    results["mixup_info"] = {'n_filtered': n_filtered, 'n_augmented': n_augment, 'n_total': n_total, 'mixup_alpha': mixup_alpha}

    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):
        model.train()
        correct, total = 0, 0
        for b_i, (x, y, sample_idxs, sample_ws) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            if before_batch_callback is not None:
                x, y, pred, sample_ws = before_batch_callback(x=x, y=y, pred=pred, sample_idxs=sample_idxs, sample_ws=sample_ws, epoch=epoch, is_eval=False)
            if pred.ndim > 2:
                if not warning_printed:
                    print(f"---\n[WARNING] pred has more than 2 dimensions: {pred.shape}.\n---")
                    warning_printed = True
                pred = pred.mean(1)
            loss = loss_fn(pred, y, step_frac=(iter_i := iter_i + 1) / n_iter, sample_weights=sample_ws)
            loss.backward()
            optimizer.step()
            schedule.step()

            y_hard = torch.argmax(y, dim=-1) if y.ndim == 2 else (y.mean(1).argmax(-1) if y.ndim == 3 else y)
            correct += (torch.argmax(pred, -1) == y_hard).detach().float().sum().item()
            total += len(y_hard)
            if after_batch_callback is not None:
                after_batch_callback(x=x, y=y_hard, pred=pred, loss=loss, last_in_epoch=b_i == n_batches - 1, epoch=epoch)

        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f} [mixup]")

        # --- Eval ---
        with torch.no_grad():
            for key, (x_ev, y_ev) in eval_datasets.items():
                x_ev = x_ev.float().to(device)
                pred = model(x_ev).detach().cpu()
                if pred.ndim > 2: pred = pred.mean(1)
                if len(pred.shape) > 1: pred = torch.argmax(pred, dim=-1)
                if len(y_ev.shape) > 1: y_ev = torch.argmax(y_ev, dim=-1)
                y_ev = y_ev.to(pred.device)
                results[f"{key}_all"].append((pred == y_ev).float().mean())

        ### print w2sg vs weak agreement/disagreement on test set every 5 epochs
        if (epoch + 1) % 5 == 0 and "test" in eval_datasets and "test_weak" in eval_datasets:
            with torch.no_grad():
                x_t, y_t = eval_datasets["val"]
                _, yw_t = eval_datasets["val_weak"]
                weak_raw_available = "val_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["val_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = model(x_t).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)

                pred_t = torch.argmax(logits_t, dim=-1)

                if weak_raw_available:
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t

                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                confidence_t = probs_t.max(dim=-1).values
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Train w2sg set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")

            with torch.no_grad():
                x_t, y_t = eval_datasets["test"]
                _, yw_t = eval_datasets["test_weak"]
                weak_raw_available = "test_weak_raw" in eval_datasets
                if weak_raw_available:
                    _, yw_t_raw = eval_datasets["test_weak_raw"]
                    if type(yw_t_raw) == np.ndarray:
                        yw_t_raw = torch.tensor(yw_t_raw, device=device)
                    elif type(yw_t_raw) == torch.Tensor:
                        yw_t_raw = yw_t_raw.to(device)

                x_t = x_t.float().to(device)
                logits_t = model(x_t).detach().cpu()
                if logits_t.ndim > 2:
                    logits_t = logits_t.mean(1)

                probs_t = torch.softmax(logits_t, dim=-1)
                log_probs_t = torch.log(probs_t + 1e-8)
                entropy_t = -(probs_t * log_probs_t).sum(dim=-1)

                pred_t = torch.argmax(logits_t, dim=-1)

                if weak_raw_available:
                    if yw_t_raw.min() >= 0 and yw_t_raw.max() <= 1.01:
                        weak_probs_t = yw_t_raw
                    else:
                        weak_probs_t = torch.softmax(yw_t_raw, dim=-1)
                    weak_confidence_t = weak_probs_t.max(dim=-1).values.detach().cpu()
                else:
                    weak_confidence_t = torch.zeros(len(y_t))

                y_t_flat = y_t.argmax(-1) if y_t.ndim > 1 else y_t
                yw_t_flat = yw_t.argmax(-1) if yw_t.ndim > 1 else yw_t

                y_t_flat = y_t_flat.cpu()
                yw_t_flat = yw_t_flat.cpu()
                w2sg_correct = (pred_t == y_t_flat)
                weak_correct = (yw_t_flat == y_t_flat)
                n_test = len(y_t_flat)

                mask_both_correct = w2sg_correct & weak_correct
                mask_w2sg_correct_weak_wrong = w2sg_correct & ~weak_correct
                mask_w2sg_wrong_weak_correct = ~w2sg_correct & weak_correct
                mask_both_wrong = ~w2sg_correct & ~weak_correct

                both_correct = mask_both_correct.float().sum().item() / n_test * 100
                w2sg_correct_weak_wrong = mask_w2sg_correct_weak_wrong.float().sum().item() / n_test * 100
                w2sg_wrong_weak_correct = mask_w2sg_wrong_weak_correct.float().sum().item() / n_test * 100
                both_wrong = mask_both_wrong.float().sum().item() / n_test * 100

                ent_both = entropy_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                ent_w2sg_right_weak_wrong = entropy_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                ent_w2sg_wrong_weak_right = entropy_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                ent_both_wrong = entropy_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                confidence_t = probs_t.max(dim=-1).values
                conf_both = confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                conf_w2sg_right_weak_wrong = confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                conf_w2sg_wrong_weak_right = confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                conf_both_wrong = confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                weak_conf_both = weak_confidence_t[mask_both_correct].mean().item() if mask_both_correct.any() else float('nan')
                weak_conf_w2sg_right_weak_wrong = weak_confidence_t[mask_w2sg_correct_weak_wrong].mean().item() if mask_w2sg_correct_weak_wrong.any() else float('nan')
                weak_conf_w2sg_wrong_weak_right = weak_confidence_t[mask_w2sg_wrong_weak_correct].mean().item() if mask_w2sg_wrong_weak_correct.any() else float('nan')
                weak_conf_both_wrong = weak_confidence_t[mask_both_wrong].mean().item() if mask_both_wrong.any() else float('nan')

                print(f"  [Epoch {epoch+1}] Test set breakdown:")
                print(f"    w2sg correct & weak correct: {both_correct:.2f}%  | avg entropy: {ent_both:.4f}  | w2sg conf: {conf_both:.4f} | weak conf: {weak_conf_both:.4f}")
                print(f"    w2sg correct & weak wrong:   {w2sg_correct_weak_wrong:.2f}%  | avg entropy: {ent_w2sg_right_weak_wrong:.4f}  | w2sg conf: {conf_w2sg_right_weak_wrong:.4f} | weak conf: {weak_conf_w2sg_right_weak_wrong:.4f}")
                print(f"    w2sg wrong   & weak correct: {w2sg_wrong_weak_correct:.2f}%  | avg entropy: {ent_w2sg_wrong_weak_right:.4f}  | w2sg conf: {conf_w2sg_wrong_weak_right:.4f} | weak conf: {weak_conf_w2sg_wrong_weak_right:.4f}")
                print(f"    w2sg wrong   & weak wrong:   {both_wrong:.2f}%  | avg entropy: {ent_both_wrong:.4f}  | w2sg conf: {conf_both_wrong:.4f} | weak conf: {weak_conf_both_wrong:.4f}")

    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]
    results['_balance_before'] = balance_before
    results['_balance_after'] = balance_after
    return results, model


# =============================================================================
# train_head_DG_mixup: DG pipeline with confidence filtering + mixup
# =============================================================================
def train_head_DG_mixup(
    teacher_model, student_model, val_dataloader, test_dataloader,
    cfg, logger, cached_labels_path, cached_embs_path, results, rng, n_classes,
    return_data=False, additional_eval_data=None,
    before_optim_run_callback_weak=None, before_optim_run_callback_gt=None,
    after_batch_callback_weak=None, before_batch_callback_weak=None,
    after_batch_callback_gt=None, before_batch_callback_gt=None,
):
    """
    train_head_DG with confidence filtering + mixup augmentation.
    Same data loading as train_head_DG_hard but uses train_logreg_mixup:
      1. Filters by weak confidence (cluster_by_confidence_v2)
      2. Mixup augmentation on embedding space with soft labels from weak model
    Extra cfg["w2s"] keys: cluster_method, weak_conf_threshold, mixup_alpha, mixup_ratio
    """
    ### get (weak) labels from current teacher
    if cfg["w2s"]["load_labels"] and os.path.exists(cached_labels_path):
        logger.info("Loading teacher labels from cache...")
        cached = torch.load(cached_labels_path, pickle_module=dill, map_location="cpu")
        val_gt_labels, val_teacher_labels = cached["val_gt_labels"], cached["val_teacher_labels"]
        test_gt_labels, test_teacher_labels = cached["test_gt_labels"], cached["test_teacher_labels"]
        teacher_acc = cached.get("teacher_acc", 0)
    else:
        chunking_dir = os.path.join(os.path.dirname(cached_labels_path), f"label_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info("Collecting teacher labels for Validation Set...")
        _, val_gt_labels, val_teacher_labels, val_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=val_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'val'))
        logger.info("Collecting teacher labels for Test Set...")
        _, test_gt_labels, test_teacher_labels, test_teacher_acc, _, _ = preload(model=partial(teacher_model, combine_logits=False, collect_embeddings=False), loader=test_dataloader, device=cfg["device"], store_embs=False, store_inps=False, chunking_dir=os.path.join(chunking_dir, 'test'))
        teacher_acc = float((np.mean(val_teacher_acc) + np.mean(test_teacher_acc)) / 2.0)
        if cfg["w2s"]["save_labels"]:
            torch.save({"cfg": cfg, "val_gt_labels": val_gt_labels, "val_teacher_labels": val_teacher_labels, "test_gt_labels": test_gt_labels, "test_teacher_labels": test_teacher_labels, "teacher_acc": teacher_acc}, cached_labels_path, pickle_module=dill)

    ### get embeddings from student model
    if cfg["w2s"]["load_embeddings"] and os.path.exists(cached_embs_path):
        logger.info("Loading student model embeddings from cache...")
        cached = torch.load(cached_embs_path, pickle_module=dill)
        val_student_embeddings, val_student_gt_labels = cached["val_embeddings"], cached["val_gt_labels"]
        test_student_embeddings, test_student_gt_labels = cached["test_embeddings"], cached["test_gt_labels"]
    else:
        chunking_dir = os.path.join(os.path.dirname(cached_embs_path), f"embs_chunks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(chunking_dir, exist_ok=True)
        logger.info("Collecting student embeddings for Validation Set...")
        val_student_embeddings, val_student_gt_labels, _, _, _, _ = preload(model=student_model, loader=val_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'val'), store_embs=True)
        logger.info("Collecting student embeddings for Test Set...")
        test_student_embeddings, test_student_gt_labels, _, _, _, _ = preload(model=student_model, loader=test_dataloader, device=cfg["device"], chunking_dir=os.path.join(chunking_dir, 'test'), store_embs=True)
        if cfg["w2s"]["save_embeddings"]:
            torch.save({"cfg": cfg, "val_embeddings": val_student_embeddings, "val_gt_labels": val_student_gt_labels, "test_embeddings": test_student_embeddings, "test_gt_labels": test_student_gt_labels}, cached_embs_path, pickle_module=dill)

    assert torch.all(val_gt_labels == val_student_gt_labels), "Val GT labels mismatch."
    assert torch.all(test_gt_labels == test_student_gt_labels), "Test GT labels mismatch."
    del val_student_gt_labels, test_student_gt_labels

    ### Shuffle and split VAL data
    order = np.arange(len(val_gt_labels))
    rng.shuffle(order)
    results["order"].append(order)
    x_val_all, y_val_all, yw_val_all = val_student_embeddings[order], val_gt_labels[order], val_teacher_labels[order]

    assert len(cfg["w2s"]["train_val_test_split_DG"]) == 2
    assert sum(cfg["w2s"]["train_val_test_split_DG"]) == 1.0
    n_train = int(cfg["w2s"]["train_val_test_split_DG"][0] * len(x_val_all))

    x_train, x_val = x_val_all[:n_train], x_val_all[n_train:]
    y_train, y_val = y_val_all[:n_train], y_val_all[n_train:]
    yw_train, yw_val = yw_val_all[:n_train], yw_val_all[n_train:]
    x_test, y_test, yw_test = test_student_embeddings, test_gt_labels, test_teacher_labels

    x = torch.cat([x_train, x_val, x_test]) if isinstance(x_train, torch.Tensor) else np.concatenate([x_train, x_val, x_test])
    y = torch.cat([y_train, y_val, y_test]) if isinstance(y_train, torch.Tensor) else np.concatenate([y_train, y_val, y_test])
    yw = torch.cat([yw_train, yw_val, yw_test]) if isinstance(yw_train, torch.Tensor) else np.concatenate([yw_train, yw_val, yw_test])

    yw_val_raw = yw_val.mean(1) if yw_val.ndim == 3 else yw_val
    yw_val = yw_val_raw.argmax(-1)
    yw_test_raw = yw_test.mean(1) if yw_test.ndim == 3 else yw_test
    yw_test = yw_test_raw.argmax(-1)

    eval_datasets = {"val": (x_val, y_val), "val_weak": (x_val, yw_val), "val_weak_raw": (x_val, yw_val_raw), "test": (x_test, y_test), "test_weak": (x_test, yw_test), "test_weak_raw": (x_test, yw_test_raw)}
    if additional_eval_data is not None:
        for k, v in additional_eval_data.items():
            eval_datasets[k] = v

    logger.info(f"\nTotal number of samples: {len(x)}.")
    logger.info(f"  Number of training samples (from Val Dataloader): {len(x_train)}.")
    logger.info(f"  Number of validation samples (from Val Dataloader): {len(x_val)}.")
    logger.info(f"  Number of testing samples (from Test Dataloader): {len(x_test)}.")

    ### eval teacher
    results["teacher_acc_src"].append(teacher_acc)
    teacher_acc_all = (y == (yw if yw.ndim == 2 else yw.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc"].append(teacher_acc_all)
    teacher_acc_train = (y_train == (yw_train if yw_train.ndim == 2 else yw_train.mean(1)).argmax(-1)).float().mean()
    results["teacher_acc_train"].append(teacher_acc_train)
    teacher_acc_val = (y_val == yw_val).float().mean()
    results["teacher_acc_val"].append(teacher_acc_val)
    teacher_acc_test = (y_test == yw_test).float().mean()
    results["teacher_acc_test"].append(teacher_acc_test)

    if type(teacher_acc) == float:
        teacher_acc = torch.tensor([teacher_acc], device=cfg["device"])
    logger.info(f"Teacher label accuracy (all data, not combined): {[np.round(tacc.item() if hasattr(tacc, 'item') else tacc, 4) for tacc in teacher_acc]}")
    logger.info(f"Teacher label accuracy (all data): {teacher_acc_all:.4f}")
    logger.info(f"Teacher label accuracy (train): {teacher_acc_train:.4f}")
    logger.info(f"Teacher label accuracy (val): {teacher_acc_val:.4f}")
    logger.info(f"Teacher label accuracy (test): {teacher_acc_test:.4f}")

    ### w2s with mixup
    if before_optim_run_callback_weak is not None:
        before_optim_run_callback_weak(yw=yw_train, sample_idxs=np.arange(len(yw_train)))
    seed_all(cfg["seed"])

    mixup_kwargs = {
        'cluster_method': cfg["w2s"].get("cluster_method", "kmeans"),
        'weak_conf_threshold': cfg["w2s"].get("weak_conf_threshold", 0.6),
        'mixup_alpha': cfg["w2s"].get("mixup_alpha", 1.0),
        'mixup_ratio': cfg["w2s"].get("mixup_ratio", 0.3),
    }
    logger.info(f"Mixup params: {mixup_kwargs}")

    results_teacher_to_student, student_model_probe = train_logreg_mixup(
        x_train, yw_train, eval_datasets, device=cfg["device"],
        batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["teacher_labels_loss_fn_name"]](**(cfg["w2s"]["teacher_labels_loss_fn_kwargs"] or dict())),
        n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None,
        before_batch_callback=before_batch_callback_weak,
        after_batch_callback=after_batch_callback_weak,
        **mixup_kwargs,
    )
    results["results_teacher_to_student"].append(results_teacher_to_student)
    results["student_model_probe"].append(student_model_probe)

    ### gt
    if before_optim_run_callback_gt is not None:
        before_optim_run_callback_gt(yw=yw_test, sample_idxs=len(y_train) + len(y_val) + np.arange(len(yw_test)))
    seed_all(cfg["seed"])
    results_gt, gt_model_probe = train_logreg(x_train, y_train, eval_datasets, device=cfg["device"], batch_size=cfg["w2s"]["batch_size"],
        loss_fn=LOSS_DICT[cfg["w2s"]["gt_labels_loss_fn_name"]](**(cfg["w2s"]["gt_labels_loss_fn_kwargs"] or dict())), n_epochs=cfg["w2s"]["n_epochs"], lr=cfg["w2s"]["lr"],
        n_classes=n_classes, sample_weights=None, before_batch_callback=before_batch_callback_gt, after_batch_callback=after_batch_callback_gt)
    results["results_gt"].append(results_gt)

    ### W2SG visualization
    _balance_stats = None
    if '_balance_before' in results_teacher_to_student and '_balance_after' in results_teacher_to_student:
        _balance_stats = {'before': results_teacher_to_student['_balance_before'], 'after': results_teacher_to_student['_balance_after']}
    if cfg["w2s"].get("plot_w2sg", False):
        _plot_dir = cfg["w2s"].get("plot_save_dir", None)
        if _plot_dir:
            _dim_methods = cfg["w2s"].get("plot_dim_reduction", ["pca"])
            if isinstance(_dim_methods, str): _dim_methods = [_dim_methods]
            for _dm in _dim_methods:
                try:
                    plot_w2sg_analysis(model=student_model_probe, gt_model=gt_model_probe, eval_datasets=eval_datasets, save_dir=_plot_dir, seed=cfg["seed"], device=cfg["device"], dim_reduction=_dm, balance_stats=_balance_stats, logger=logger)
                except Exception as _plot_err:
                    logger.info(f"[W2SG Plot] Warning: plotting failed ({_dm}): {_plot_err}")

    if return_data:
        return results, student_model_probe, {"x": x, "y": y, "yw": yw, "x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val, "x_test": x_test, "y_test": y_test, "yw_train": yw_train, "yw_val": yw_val, "yw_test": yw_test}
    return results, student_model_probe
