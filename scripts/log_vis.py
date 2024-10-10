import json
import gzip
import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--episodes_file', type=str, default="data/datasets/eqa/mp3d/v1/val/val.json.gz")
    parser.add_argument('--log_file', type=str, default="results/models/eqa_baseline/log.json")
    args = parser.parse_args()

    episodes_file=args.episodes_file
    log_file=args.log_file
    with gzip.open(episodes_file, 'r') as f:
        episodes_data = json.loads(
            f.read().decode('utf-8'))

    with open(log_file, 'r') as f:
        _log_data = json.loads(
            f.read())

    log_data= []
    for scene, scene_dct in _log_data.items():
        for i, dct in scene_dct.items():
            log_data.append(dct)
    print(len(log_data), len(episodes_data["episodes"]))

    eps_num = 0
    vqa_acc = 0
    for dct in log_data:
        if dct["invalid"]==False:
            eps_num += 1
            if dct["vqa_success"]:
                vqa_acc += 1
    vqa_acc /= eps_num
    print(f"VQA accuracy: {vqa_acc} ({eps_num})")

    # metrics for navigation
    eps_num = 0
    d_T = 0
    d_D = 0
    d_min = 0
    d_0 = 0
    for dct in log_data:
        if dct["invalid"]==False:
            eps_num += 1
            item = dct["pacman"]
            d_T += item["d_T"]
            d_D += item["d_D"]
            d_min += item["d_min"]
            d_0 += item["d_0"]
    d_T /= eps_num
    d_D /= eps_num
    d_min /= eps_num
    d_0 /= eps_num
    print(f"d_T: {d_T} ({eps_num})")
    print(f"d_D: {d_D} ({eps_num})")
    print(f"d_min: {d_min} ({eps_num})")
    print(f"d_0: {d_0} ({eps_num})")



if __name__=="__main__":
    main()
