import json
import bz2
import gzip
import os, sys
import numpy as np
import torch
import argparse

np.random.seed(seed=0)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--apikey_path', type=str, default=None, required=True)
    parser.add_argument('--split', choices=["train", "val"], help='Choose dataset split to preprocess')
    parser.add_argument('--dataset', type=str, default="mp3d", required=True)
    parser.add_argument('--version', type=str, default="v1", required=True)
    parser.add_argument('--input_eqa_dataset', type=str, default=None, required=True)
    args = parser.parse_args()
    split = "train"
    obs_point = "end"
    dataset = "mp3d"
    version="v_zeroshot"
    save_path = f"data/datasets/eqa/{dataset}/{version}/{split}/scene_split/"
    episodes_file=f"data/datasets/eqa/{dataset}/{version}/{split}/{split}.json.gz"
    train_val_split_ratio=0.8

    with gzip.open(episodes_file, 'r') as f:
        original_data = json.loads(
            f.read().decode('utf-8'))
        json_data = original_data["episodes"]
        habitat_data = original_data.copy()

    # extract scene_ids
    scene_ids_all = []
    for idx, episode in enumerate(json_data):
        if episode["scene_id"] not in scene_ids_all:
            scene_ids_all.append(episode["scene_id"])
    
    #train-val splitter
    scene_indices = np.arange(0, len(scene_ids_all))
    np.random.shuffle(scene_indices)
    train_scene_indices = scene_indices[:int(train_val_split_ratio*len(scene_ids_all))]
    val_scene_indices = scene_indices[int(train_val_split_ratio*len(scene_ids_all)):]
    
    train_scene_ids = []
    for scene_index in train_scene_indices:
        train_scene_ids.append(scene_ids_all[scene_index])
    val_scene_ids = []
    for scene_index in val_scene_indices:
        val_scene_ids.append(scene_ids_all[scene_index])

    # Habitat data
    train_data_habitat = habitat_data.copy()
    val_data_habitat = habitat_data.copy()
    train_data_habitat["episodes"] = []
    val_data_habitat["episodes"] = []
    for episode in habitat_data["episodes"]:
        scene_id = episode["scene_id"]
        if scene_id in train_scene_ids:
            train_data_habitat["episodes"].append(episode)
        elif scene_id in val_scene_ids:
            val_data_habitat["episodes"].append(episode)


    save_file_path_habitat_train = os.path.join(save_path, "train_train")
    save_file_path_habitat_val = os.path.join(save_path, "train_val")

    json_data_save = json.dumps(train_data_habitat, indent=2)
    # Convert to bytes
    encoded = json_data_save.encode('utf-8')
    # Compress
    compressed = gzip.compress(encoded)
    with open(save_file_path_habitat_train +".json.gz", "wb") as f:
        f.write(compressed)

    json_data_save = json.dumps(val_data_habitat, indent=2)
    # Convert to bytes
    encoded = json_data_save.encode('utf-8')
    # Compress
    compressed = gzip.compress(encoded)
    with open(save_file_path_habitat_val +".json.gz", "wb") as f:
        f.write(compressed)


if __name__ == "__main__":
    main()
