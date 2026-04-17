"""
Script to parse w2sg vs weak breakdown from log files across 3 seeds,
average the results, and save to a summary log file.

Usage:
    python parse_logs.py /home/adminn/theanh28/raven/rw2s/logs/run_w2s/pacs_ens_3
    
    # Or process ALL experiment folders at once:
    python parse_logs.py /home/adminn/theanh28/raven/rw2s/logs/run_w2s
"""

import os
import re
import sys


# --------------------------------------------------------------------------
# Keys for the 4 breakdown cases
# --------------------------------------------------------------------------
CASE_KEYS = [
    "both_correct",
    "w2sg_correct_weak_wrong",
    "w2sg_wrong_weak_correct",
    "both_wrong",
]

# Display labels matching the log format
CASE_LABELS = {
    "both_correct":              "w2sg correct & weak correct:",
    "w2sg_correct_weak_wrong":   "w2sg correct & weak wrong:  ",
    "w2sg_wrong_weak_correct":   "w2sg wrong   & weak correct:",
    "both_wrong":                "w2sg wrong   & weak wrong:  ",
}


def _build_block_pattern(header_re):
    """
    Build a regex that matches a breakdown header followed by 4 metric lines.
    Each metric line has: percentage | avg entropy | w2sg conf | weak conf
    Returns compiled regex with 16 groups (4 lines × 4 values).
    """
    metric_line = (
        r"\s*w2sg (?:correct|wrong)\s*& weak (?:correct|wrong)\s*:\s*"
        r"([\d.]+)%\s*\|\s*avg entropy:\s*([\d.]+)\s*\|\s*w2sg conf:\s*([\d.]+)\s*\|\s*weak conf:\s*([\d.]+)"
    )
    full = header_re + r"\s*\n" + r"\s*\n".join([metric_line] * 4)
    return re.compile(full)


# Patterns for Train w2sg and Test breakdowns at any epoch
TRAIN_PATTERN = _build_block_pattern(r"\[Epoch (\d+)\] Train w2sg set breakdown:")
TEST_PATTERN  = _build_block_pattern(r"\[Epoch (\d+)\] Test set breakdown:")


def _extract_case_dict(match, group_offset):
    """
    From a regex match, extract one case's metrics starting at group_offset.
    Returns dict with pct, entropy, w2sg_conf, weak_conf.
    """
    return {
        "pct":       float(match.group(group_offset)),
        "entropy":   float(match.group(group_offset + 1)),
        "w2sg_conf": float(match.group(group_offset + 2)),
        "weak_conf": float(match.group(group_offset + 3)),
    }


def _parse_blocks(content, pattern):
    """
    Parse all breakdown blocks matching `pattern` from file content.
    Returns list of (epoch, {case_key: {pct, entropy, w2sg_conf, weak_conf}}).
    """
    results = []
    for m in pattern.finditer(content):
        epoch = int(m.group(1))
        cases = {}
        for i, key in enumerate(CASE_KEYS):
            # group 1 = epoch, then groups 2-5 = case 0, 6-9 = case 1, ...
            offset = 2 + i * 4
            cases[key] = _extract_case_dict(m, offset)
        results.append((epoch, cases))
    return results


def parse_log_file(filepath):
    """
    Parse a single log file. Returns dict with:
      "train_blocks": [(epoch, cases), ...]
      "test_blocks":  [(epoch, cases), ...]
    Each `cases` is {case_key: {pct, entropy, w2sg_conf, weak_conf}}.
    """
    with open(filepath, "r") as f:
        content = f.read()

    return {
        "train_blocks": _parse_blocks(content, TRAIN_PATTERN),
        "test_blocks":  _parse_blocks(content, TEST_PATTERN),
    }


def _get_last_epoch_block(blocks):
    """Return the cases dict from the block with the highest epoch number."""
    if not blocks:
        return None
    return max(blocks, key=lambda x: x[0])  # (epoch, cases)


def _avg_cases(case_list):
    """
    Average a list of cases dicts across seeds.
    Returns {case_key: {pct, entropy, w2sg_conf, weak_conf}} with averaged values.
    """
    n = len(case_list)
    avg = {}
    for key in CASE_KEYS:
        avg[key] = {
            "pct":       sum(c[key]["pct"] for c in case_list) / n,
            "entropy":   sum(c[key]["entropy"] for c in case_list) / n,
            "w2sg_conf": sum(c[key]["w2sg_conf"] for c in case_list) / n,
            "weak_conf": sum(c[key]["weak_conf"] for c in case_list) / n,
        }
    return avg


def process_experiment(exp_dir):
    """
    Process one experiment folder (e.g. pacs_ens_3).
    Reads seed0.log, seed1.log, seed2.log from the logs/ subfolder.
    Returns dict with averaged results for weak/gt training on both train and test sets.
    """
    logs_dir = os.path.join(exp_dir, "logs")
    if not os.path.isdir(logs_dir):
        print(f"  [SKIP] No 'logs/' subfolder in {exp_dir}")
        return None

    seed_files = [os.path.join(logs_dir, f"seed{i}.log") for i in range(3)]
    missing = [f for f in seed_files if not os.path.exists(f)]
    if missing:
        print(f"  [SKIP] Missing seed files: {[os.path.basename(f) for f in missing]}")
        return None

    # Parse all 3 seeds
    all_seeds = []
    for sf in seed_files:
        parsed = parse_log_file(sf)
        all_seeds.append(parsed)

    # For each seed, the log may contain multiple training runs (weak label, gt label).
    # The FIRST set of breakdown blocks = weak-label training.
    # The SECOND set = gt-label training.
    # We group consecutive blocks by looking at epoch resets.

    results = {}

    for block_type, set_name in [("train_blocks", "train"), ("test_blocks", "test")]:
        for run_idx, run_label in enumerate(["weak_label", "gt_label"]):
            key = f"{run_label}_{set_name}"

            seed_cases = []
            for seed_data in all_seeds:
                blocks = seed_data[block_type]
                # Split blocks into runs: when the same block_type appears twice,
                # the first set belongs to weak-label training, second to gt-label.
                # A simple heuristic: split into runs where epoch resets (decreases).
                runs = _split_into_runs(blocks)
                if run_idx < len(runs) and runs[run_idx]:
                    # Take the last epoch from this run
                    last = max(runs[run_idx], key=lambda x: x[0])
                    seed_cases.append(last[1])

            if not seed_cases:
                continue

            n = len(seed_cases)
            avg = _avg_cases(seed_cases)
            # Get epoch number from any seed's last block
            sample_blocks = None
            for seed_data in all_seeds:
                runs = _split_into_runs(seed_data[block_type])
                if run_idx < len(runs) and runs[run_idx]:
                    sample_blocks = runs[run_idx]
                    break
            last_epoch = max(b[0] for b in sample_blocks) if sample_blocks else "?"

            results[key] = {
                "avg": avg,
                "n_seeds": n,
                "per_seed": seed_cases,
                "last_epoch": last_epoch,
            }

    return results


def _split_into_runs(blocks):
    """
    Split a list of (epoch, cases) blocks into separate runs.
    A new run starts when the epoch number is <= the previous epoch.
    """
    if not blocks:
        return []
    runs = [[blocks[0]]]
    for i in range(1, len(blocks)):
        if blocks[i][0] <= blocks[i - 1][0]:
            runs.append([])
        runs[-1].append(blocks[i])
    return runs


def format_results(exp_name, results):
    """Format results as a readable string."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"Experiment: {exp_name}")
    lines.append(f"{'=' * 70}")

    for run_label, display_name in [("weak_label", "Trained with WEAK labels"),
                                     ("gt_label", "Trained with GT labels")]:
        # Check if we have any data for this run
        has_data = any(f"{run_label}_{s}" in results for s in ["train", "test"])
        if not has_data:
            continue

        lines.append(f"\n--- {display_name} ---")

        for set_name, set_display in [("train", "Train w2sg set"), ("test", "Test set")]:
            key = f"{run_label}_{set_name}"
            if key not in results:
                continue

            data = results[key]
            avg = data["avg"]
            n = data["n_seeds"]
            epoch = data["last_epoch"]

            lines.append(f"\n  [Epoch {epoch}] {set_display} breakdown (avg over {n} seeds):")
            for case_key in CASE_KEYS:
                c = avg[case_key]
                label = CASE_LABELS[case_key]
                lines.append(
                    f"    {label} {c['pct']:6.2f}%  "
                    f"| avg entropy: {c['entropy']:.4f}  "
                    f"| w2sg conf: {c['w2sg_conf']:.4f} "
                    f"| weak conf: {c['weak_conf']:.4f}"
                )

            # Per-seed detail
            lines.append(f"  Per-seed details ({set_display}):")
            for i, sb in enumerate(data["per_seed"]):
                parts = []
                for case_key in CASE_KEYS:
                    short = case_key.replace("w2sg_correct_weak_wrong", "w2sg✓_weak✗") \
                                    .replace("w2sg_wrong_weak_correct", "w2sg✗_weak✓") \
                                    .replace("both_correct", "both✓") \
                                    .replace("both_wrong", "both✗")
                    parts.append(f"{short}={sb[case_key]['pct']:.2f}%")
                lines.append(f"    seed{i}: {', '.join(parts)}")

    lines.append(f"\n{'=' * 70}\n")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_logs.py <experiment_dir_or_parent_dir>")
        print("  e.g. python parse_logs.py /home/adminn/theanh28/raven/rw2s/logs/run_w2s/pacs_ens_3")
        print("  e.g. python parse_logs.py /home/adminn/theanh28/raven/rw2s/logs/run_w2s  (processes all)")
        sys.exit(1)

    input_path = sys.argv[1]

    # Determine if this is a single experiment or a parent containing many
    def is_experiment_dir(path):
        return os.path.isfile(os.path.join(path, "logs", "seed0.log"))

    if is_experiment_dir(input_path):
        child_exp_dirs = sorted([
            os.path.join(input_path, d)
            for d in os.listdir(input_path)
            if os.path.isdir(os.path.join(input_path, d)) and is_experiment_dir(os.path.join(input_path, d))
        ])
        if child_exp_dirs:
            exp_dirs = child_exp_dirs
        else:
            exp_dirs = [input_path]
    else:
        exp_dirs = sorted([
            os.path.join(input_path, d)
            for d in os.listdir(input_path)
            if os.path.isdir(os.path.join(input_path, d)) and is_experiment_dir(os.path.join(input_path, d))
        ])

    if not exp_dirs:
        print(f"No experiment folders found in {input_path}")
        sys.exit(1)

    all_output = []
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        print(f"Processing: {exp_name} ...")
        results = process_experiment(exp_dir)
        if results:
            formatted = format_results(exp_name, results)
            all_output.append(formatted)
            print(formatted)
        else:
            print(f"  [SKIP] Could not process {exp_name}\n")

    # Save summary to file
    if all_output:
        output_path = os.path.join(input_path, "epoch30_summary_all.log")
        if len(exp_dirs) == 1:
            output_path = os.path.join(exp_dirs[0], "epoch30_summary_all.log")

        with open(output_path, "w") as f:
            f.write(f"W2SG vs Weak Breakdown Summary\n")
            f.write(f"(Averaged over 3 seeds: seed0, seed1, seed2)\n\n")
            f.write("\n".join(all_output))

        print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()
