#!/usr/bin/env bash
set -euo pipefail

# bash run_w2s_multi_seed.sh pacs_test_raven
MODEL=$1

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_CFG="$BASE_DIR/../configs/${MODEL}.yaml"
OUTPUT_ROOT="$BASE_DIR/../logs/run_w2s/${MODEL}"
CONFIG_OUTPUT_DIR="$OUTPUT_ROOT/configs"
LOG_OUTPUT_DIR="$OUTPUT_ROOT/logs"
IMAGE_OUTPUT_DIR="$OUTPUT_ROOT/images"

mkdir -p "$CONFIG_OUTPUT_DIR"
mkdir -p "$LOG_OUTPUT_DIR"
mkdir -p "$IMAGE_OUTPUT_DIR"

# Seeds to run sequentially for the same config.
SEEDS=(0 1 2)

# Use this if run_w2s.py requires a working directory inside the vision folder.
cd "$BASE_DIR"

if [[ -z "${MODEL:-}" ]]; then
  echo "Usage: $0 <config-name-without-yaml>"
  echo "Example: $0 pacs_test_raven"
  exit 1
fi

for seed in "${SEEDS[@]}"; do
  run_name="seed${seed}"
  output_cfg="$CONFIG_OUTPUT_DIR/${run_name}.yaml"
  log_file="$LOG_OUTPUT_DIR/${run_name}.log"

  echo "Generating config for ${run_name}..."
  python3 - "$TEMPLATE_CFG" "$output_cfg" "$seed" "$IMAGE_OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path
from ruamel.yaml import YAML

template_cfg = Path(sys.argv[1])
output_cfg = Path(sys.argv[2])
seed = int(sys.argv[3])
image_dir = sys.argv[4]

yaml = YAML()
with template_cfg.open('r') as f:
    cfg = yaml.load(f)

cfg['seed'] = seed

# Inject plot_save_dir so images are saved alongside logs
if 'w2s' in cfg and cfg['w2s'] is not None:
    cfg['w2s']['plot_save_dir'] = image_dir

with output_cfg.open('w') as f:
    yaml.dump(cfg, f)
PY

  echo "Running experiment ${run_name}..."
  PYTHONPATH="$BASE_DIR/../..:${PYTHONPATH:-}" python3 "$BASE_DIR/run_w2s.py" --cfg_path "$output_cfg" 2>&1 | tee "$log_file"

  echo "Saved config: $output_cfg"
  echo "Saved log: $log_file"
  echo "----------------------------------------"
done

echo "All runs complete. Configs saved to $CONFIG_OUTPUT_DIR and logs saved to $LOG_OUTPUT_DIR."
