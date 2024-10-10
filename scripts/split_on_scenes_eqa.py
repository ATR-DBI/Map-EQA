import os, sys
import gzip
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--split', choices=["train", "val"], help='Choose dataset split to preprocess')
    parser.add_argument('--dataset', type=str, default="mp3d", required=True)
    parser.add_argument('--version', type=str, default="v1", required=True)
    parser.add_argument('--input_eqa_dataset', type=str, default=None, required=True)
    args = parser.parse_args()

    split = args.split
    version = args.version
    dataset = args.dataset
    episodes_file = args.input_eqa_dataset
    output_path = f"data/datasets/eqa/{dataset}/{version}/{split}/content"

    with gzip.open(episodes_file, 'r') as f:
        json_data = json.loads(
            f.read().decode('utf-8'))

    episodes_per_scene={}
    for episode in json_data["episodes"]:
        scene_name = episode["scene_id"].split("/")[1]
        if scene_name not in episodes_per_scene.keys():
            episodes_per_scene[scene_name] = {}
            episodes_per_scene[scene_name]["episodes"] = []
        episodes_per_scene[scene_name]["episodes"].append(episode)

    for scene in episodes_per_scene.values():
        scene['question_vocab'] = json_data['question_vocab']
        scene['answer_vocab'] = json_data['answer_vocab']

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for scene_name, scene_data in episodes_per_scene.items():
        file_path = os.path.join(output_path, f"{scene_name}.json.gz")
        json_data = json.dumps(scene_data, indent=2)
        # Convert to bytes
        encoded = json_data.encode('utf-8')
        # Compress
        compressed = gzip.compress(encoded)
        with open(file_path, "wb") as f:
            f.write(compressed)


if __name__=="__main__":
    main()