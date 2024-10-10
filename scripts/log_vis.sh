EPISODES_FILE="data/datasets/eqa/mp3d/v1/val/val.json.gz"
LOG_FILE="results/models/eqa_baseline/log.json"

python scripts/log_vis.py \
    --episodes_file $EPISODES_FILE \
    --log_file $LOG_FILE
