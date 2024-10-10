
# set the apikey_path correctly
python scripts/add_llm_captions_to_dataset_azure.py \
    --apikey_path ./data/*.json \
    --split val \
    --dataset mp3d \
    --version v1 \
    --input_eqa_dataset ./data/val.json.gz

python scripts/split_on_scenes_eqa.py \
    --split val \
    --dataset mp3d \
    --version v1 \
    --input_eqa_dataset ./data/val.json.gz
