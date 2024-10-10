from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np
import gzip

from envs import make_vec_eqa_vqa_envs
from arguments_eqa import get_args
import skimage.morphology

from PIL import Image
from lavis.models import load_model_and_preprocess, load_model
from lavis.processors import load_processor

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print(args)
    

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    episode_vqa_success = []
    for _ in range(args.num_processes):
        episode_vqa_success.append(deque(maxlen=num_episodes))

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_eqa_vqa_envs(args)
    obs, infos = envs.reset()

    torch.set_grad_enabled(False)

    # Global policy observation space
    ngc = int(8 + args.num_sem_categories)

    # Global policy recurrent layer size
    g_hidden_size = args.global_hidden_size


    # Global policy
    episodes_file="data/datasets/eqa/mp3d/v1/train/train.json.gz"
    with gzip.open(episodes_file, 'r') as f:
        json_data = json.loads(
            f.read().decode('utf-8'))
        q_vocab = json_data['question_vocab']["word2idx_dict"]
        answer_candidates = list(json_data["answer_vocab"]["word2idx_dict"].keys())
    
    q_rnn_kwargs = {
        "token_to_idx": q_vocab,
        'num_sem_categories': ngc - 8,
        'recurrent': args.use_recurrent_global,
        'hidden_size': g_hidden_size,
    }
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
    model_vqa, vis_processors_vqa, text_processors_vqa = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

    if args.load_finetuned:
        model_vqa = load_model("blip_vqa", model_type="vqav2", is_eval=True, device=device, checkpoint=args.vqa_chpt_path)

    # obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

    start = time.time()

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    save_data={}
    finished_episode_ids = []
    
    print(args.num_training_frames // args.num_processes + 1)
    for step in range(args.num_training_frames // args.num_processes + 1):
        print(finished)
        if finished.sum() == args.num_processes:
            print("BREAK")
            break

        planner_inputs =[{} for i in range(num_scenes)]
        for e in range(num_scenes):
            planner_inputs[e]["wait"] = finished[e]
        obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

        # VQA
        pred_answers = [0 for i in range(num_scenes)]
        vqa_success=[0 for i in range(num_scenes)]
        # __obs = envs.get_obs()
        itm_scores = [0 for i in range(num_scenes)]
        # For every global step, update the full and local maps
        for e in range(num_scenes):
            # pano = __obs[e]["rgb_equirectangular"]
            # # caption = infos[e]["llm_declarative_text"]
            # caption = infos[e]["question_text"]
            # question = infos[e]["question_text"]
            # answer = infos[e]["answer"]
            itm_scores[e] = 0

            pred_answers[e] = 0
            vqa_success[e] = 0
            infos[e]["vqa_success"] = 0
            infos[e]["itm_score"] = 0
            
        
        # update = False
        for e, x in enumerate(done):
            if x:
                # vqa_success = infos[e]['vqa_success']
                # if args.eval:
                    
                # if len(episode_success[e]) == num_episodes:
                if infos[e]["thread_finished"]:
                    finished[e] = 1
                if infos[e]["scene_name"] not in save_data.keys():
                    save_data[infos[e]["scene_name"]]={}
                scene_name=infos[e]["scene_name"]
                if infos[e]["episode_id"] not in finished_episode_ids:
                    save_data[infos[e]["scene_name"]][infos[e]["episode_id"]] = infos[e]
                    save_data[infos[e]["scene_name"]][infos[e]["episode_id"]]["question_token"] = infos[e]["question_token"].tolist()
                    finished_episode_ids.append(infos[e]["episode_id"])
                    episode_vqa_success[e].append(vqa_success[e])
                    update = True
        # print(finished)
        # if update:
        #     with open(log_filename, "w") as f:
        #         json.dump(save_data,f, indent=4)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging
        if step % args.log_interval == 0:
            print(f"Timestep: {(step * num_scenes)}")
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
    # Logging
    if step % args.log_interval == 0:
        end = time.time()
        time_elapsed = time.gmtime(end - start)
        log = " ".join([
            "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
            "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
            "num timesteps {},".format(step * num_scenes),
            "FPS {},".format(int(step * num_scenes / (end - start)))
        ])


        if args.eval:
            total_vqa_success = []
            for e in range(args.num_processes):
                for vqa_acc in episode_vqa_success[e]:
                    total_vqa_success.append(vqa_acc)
                log += " ObjectNav vqa_succ:"
                log += " {:.3f}({:.0f}),".format(
                    np.mean(total_vqa_success),
                    len(total_vqa_success))
        print(log)
    
    # with open(log_filename, "w") as f:
    #     json.dump(save_data,f, indent=4)


if __name__ == "__main__":
    main()
