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

from model import RL_Policy, Semantic_Mapping
from utils.storage import GlobalRolloutStorage
from envs import make_vec_eqa_envs
from arguments_eqa import get_args
import skimage.morphology
# from agents.policy.discrete_planner import DiscretePlanner

from PIL import Image
from lavis.models import load_model_and_preprocess, load_model
from lavis.processors import load_processor
import envs.utils.pose as pu
import math

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    args = get_args()
    if args.eval==0:
        raise ValueError("Only Evaluation is available!")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    append = args.back_step_num
    if args.back_step_num in ["10", "30", "50"]:
            append = f"T_{args.back_step_num}_sem{str(args.detic_confidence_threshold)}_itm{str(args.text_img_matching_threshold)}_stop_dist_{str(args.stg_stop_dist)}"
    else:
        append = f"{args.back_step_num}_sem{str(args.detic_confidence_threshold)}_itm{str(args.text_img_matching_threshold)}_stop_dist_{str(args.stg_stop_dist)}"
    log_dir = "results/dump/{}/{}/{}/{}/".format(args.exp_name, args.dataset, args.dataset_version, append)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    
    # save args as a file
    args_config = vars(args)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({'parser_config': args_config}, f, indent=4)
    ###############

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)
    log_filename=log_dir + 'log.json'

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    g_masks = torch.ones(num_scenes).float().to(device)

    best_g_reward = -np.inf

    if args.eval:
        episode_success = []
        episode_spl = []
        episode_dist = []
        episode_vqa_success = []
        for _ in range(args.num_processes):
            episode_success.append(deque(maxlen=num_episodes))
            episode_spl.append(deque(maxlen=num_episodes))
            episode_dist.append(deque(maxlen=num_episodes))
            episode_vqa_success.append(deque(maxlen=num_episodes))

    else:
        episode_success = deque(maxlen=1000)
        episode_spl = deque(maxlen=1000)
        episode_dist = deque(maxlen=1000)

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)
    g_dist_losses = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_eqa_envs(args)
    obs, infos = envs.reset()

    torch.set_grad_enabled(False)

    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = int(args.num_sem_categories + 4)  # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution # 各pixelはdefaultで5cm、map_size_cmは2400cm, 
    full_w, full_h = map_size, map_size# defaultでmap_size=480 pixel
    local_w = int(full_w / args.global_downscaling) # defaultで1/2にdownscaleするので、local_w,local_hは240
    local_h = int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:# full_wをoverしないように設定
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0 # m換算で、マップの中心位置を指すと思うが、map_resolutionの値

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]#pixel単位のいちに直す。ここで代入されるのはマップの中心pixelのいち

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]#マップの左上の座標のこと

        for e in range(num_scenes):
            local_map[e] = full_map[e, :,
                                    lmb[e, 0]:lmb[e, 1],
                                    lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                torch.from_numpy(origins[e]).to(device).float()

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                          (local_w, local_h),
                                          (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()
        llm_category_masks[e].fill_(1.)

    def update_intrinsic_rew(e):
        prev_explored_area = full_map[e, 1].sum(1).sum(0)
        full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
            local_map[e]
        curr_explored_area = full_map[e, 1].sum(1).sum(0)
        intrinsic_rews[e] = curr_explored_area - prev_explored_area
        intrinsic_rews[e] *= (args.map_resolution / 100.)**2  # to m^2

    init_map_and_pose()

    # Global policy observation space
    ngc = int(8 + args.num_sem_categories)
    es = 1+1 # direction and LLM pred object 
    g_observation_space = gym.spaces.Box(0, 1,
                                         (ngc,
                                          local_w,
                                          local_h), dtype='uint8')

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=0.99,
                                    shape=(2,), dtype=np.float32)

    # Global policy recurrent layer size
    g_hidden_size = args.global_hidden_size

    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()

    # Global policy
    
    episodes_file="data/datasets/eqa/mp3d/v1/train/train.json.gz"
    with gzip.open(episodes_file, 'r') as f:
        json_data = json.loads(
            f.read().decode('utf-8'))
        q_vocab = json_data['question_vocab']["word2idx_dict"]
        answer_candidates = list(json_data["answer_vocab"]["word2idx_dict"].keys())
    
    from agents.policy.nf_exp import NearestFrontierExp
    from agents.policy.nf_planner import PlannerActor
    g_policy = NearestFrontierExp(args)

    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    intrinsic_rews = torch.zeros(num_scenes).to(device)
    extras = torch.zeros(num_scenes, 2) # direction and LLM pred object 

    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, g_observation_space.shape,
                                      g_action_space, g_policy.rec_state_size,
                                      es).to(device)
    # model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)
    model_vqa, vis_processors_vqa, text_processors_vqa = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

    if args.load_finetuned:
        model_vqa = load_model("blip_vqa", model_type="vqav2", is_eval=True, device=device, checkpoint=args.vqa_chpt_path)

    if args.eval:
        g_policy.eval()

    # Predict semantic map from frame 1
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
    ).float().to(device)
    poses[:,0] = - poses[:, 0]

    _, local_map, _, local_pose = \
        sem_map_module(obs, poses, local_map, local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
    global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
        full_map[:, 0:4, :, :])
    global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()

    for env_idx in range(num_scenes):
        if env_idx==0:
            llm_objects = infos[env_idx]["llm_object_id"].unsqueeze(0)
        else:
            llm_objects = torch.cat((llm_objects, infos[env_idx]["llm_object_id"].unsqueeze(0)), 0)

    extras = torch.zeros(num_scenes, 2)
    extras[:, 0] = global_orientation[:, 0]
    extras[:, 1] = llm_objects

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(extras)
    # Get Long-term goal 
    # Compute additional inputs if needed
    extra_maps = {
        "dmap": None,
        "umap": None,
        "fmap": None,
        "pfs": None,
        "agent_locations": None,
        "ego_agent_poses": None,
    }

    extra_maps["agent_locations"] = []
    for e in range(num_scenes):
        pose_pred = planner_pose_inputs[e]
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        map_r, map_c = start_y, start_x
        map_loc = [
            int(map_r * 100.0 / args.map_resolution - gx1),
            int(map_c * 100.0 / args.map_resolution - gy1),
        ]
        map_loc = pu.threshold_poses(map_loc, global_input[e].shape[1:])
        extra_maps["agent_locations"].append(map_loc)

    # get frontier map
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        obs_map = local_map[e, 0, :, :].cpu().numpy()
        exp_map = local_map[e, 1, :, :].cpu().numpy()
        p_input["obs_map"] = obs_map
        p_input["exp_map"] = exp_map
        p_input["pose_pred"] = planner_pose_inputs[e]
    masks = [1 for _ in range(num_scenes)]
    # fmap = nf_planner.get_frontier_maps(planner_inputs, masks)
    fmap = envs.get_frontier_map(planner_inputs)
    # import pdb;pdb.pdb.set_trace()
    extra_maps["fmap"] = torch.from_numpy(fmap).to(device)

    if g_policy.needs_egocentric_transform:
        ego_agent_poses = []
        for e in range(num_scenes):
            map_loc = extra_maps["agent_locations"][e]
            # Crop map about a center
            ego_agent_poses.append(
                [map_loc[0], map_loc[1], math.radians(start_o)]
            )
        ego_agent_poses = torch.Tensor(ego_agent_poses).to(device)
        extra_maps["ego_agent_poses"] = ego_agent_poses

    ####################################################################
    # Sample long-term goal from global policy
    _, g_action, _, _ = g_policy.act(
        global_input,
        None,
        g_masks,
        extras=extras.long(),
        deterministic=False,
        extra_maps=extra_maps,
    )
    
    cpu_actions = g_action.cpu().numpy()
    if len(cpu_actions.shape) == 4:
        # Output action map
        global_goals = cpu_actions[:, 0]  # (B, H, W)
    elif len(cpu_actions.shape) == 3:
        # Output action map
        global_goals = cpu_actions  # (B, H, W)
    
    
    g_masks.fill_(1.0)
    ########################################################################
    # Define long-term goal map
    ########################################################################
    goal_maps = [
        np.zeros((local_w, local_h)) for _ in range(num_scenes)
    ]
    for e in range(num_scenes):
        goal_maps[e][:, :] = global_goals[e]


    # Update long-term goal if target object is found
    found_goal = [0 for _ in range(num_scenes)]

    for e in range(num_scenes):
        cn = infos[e]['llm_object_id'] + 4
        if local_map[e, cn, :, :].sum() >= 1.:
            print(f"Thread {e} found goal (ch.{cn}), sum: {local_map[e, cn, :, :].sum()}")
            cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
            cat_semantic_scores = cat_semantic_map
            cat_semantic_scores[cat_semantic_scores > 0] = 1.
            goal_maps[e] = cat_semantic_scores
            found_goal[e] = 1

    
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e]  # global_goals[e]
        p_input['new_goal'] = 1
        p_input['found_goal'] = found_goal[e]
        p_input['wait'] = wait_env[e] or finished[e]
        p_input["goal_confident"]=False
        p_input["itm_done"] = False
        p_input["itm_score"] = 0
        p_input["fmap"] = fmap[e] # to save map as png
        if args.visualize or args.print_images:
            tmp_local_map = local_map[e].detach().clone()
            tmp_local_map[-1,:,:] = 1e-5
            p_input['sem_map_pred'] = tmp_local_map[4:, :, :
                                                ].argmax(0).cpu().numpy()
    # import pdb;pdb.pdb.set_trace() # planner_inputs[0]["sem_map_pred"]
    _planner_inputs = planner_inputs
    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

    start = time.time()
    g_reward = 0

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)


    selem = skimage.morphology.disk(int(5)) # そのobjectの中心を向くために、そこまで大きい値でなくてよい.
    # selem = skimage.morphology.disk(
    #             int(args.mask_dist* 100. / args.map_resolution))
    llm_category_masks = []
    for e in range(num_scenes):
        llm_category_mask = (full_map[e,int(llm_objects[e] + 4)]).to('cpu').detach().numpy().copy().astype(int)
        llm_category_mask = skimage.morphology.binary_dilation(
            llm_category_mask, selem) != True # objectが観測されたところは0に. 観測されていないところは1に.
        llm_category_mask = torch.from_numpy(llm_category_mask.astype(np.float32)).clone()
        llm_category_masks.append(llm_category_mask)
    # Initialization
    goal_confidence = []
    for e in range(num_scenes):
        goal_confidence.append(False)
    itm_done = []
    for e in range(num_scenes):
        itm_done.append(False)
    save_data={}
    finished_episode_ids = []

    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break
        __done_ = []
        for e in range(num_scenes):
            if infos[e]["thread_finished"]:
                __done_.append(1)
            else:
                __done_.append(0)
        __done_ = np.array(__done_)
        print(finished, __done_, args.num_processes)
        # print(step)
        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1
                                     for x in done]).to(device)
        g_masks *= l_masks
        UPDATED=False
        for e, x in enumerate(done):
            if x:
                spl = infos[e]['spl']
                success = infos[e]['success']
                dist = infos[e]['distance_to_goal']
                spl_per_category[infos[e]['goal_name']].append(spl)
                success_per_category[infos[e]['goal_name']].append(success)
                vqa_success = infos[e]['vqa_success']
                if args.eval:
                    episode_success[e].append(success)
                    episode_spl[e].append(spl)
                    episode_dist[e].append(dist)
                    episode_vqa_success[e].append(vqa_success)
                    # if len(episode_success[e]) == num_episodes:
                    if infos[e]["thread_finished"]:
                        finished[e] = 1
                    if infos[e]["scene_name"] not in save_data.keys():
                        save_data[infos[e]["scene_name"]]={}
                    scene_name=infos[e]["scene_name"]
                    if infos[e]["episode_id"] not in finished_episode_ids:
                        UPDATED=True
                        save_data[infos[e]["scene_name"]][infos[e]["episode_id"]] = infos[e]
                        save_data[infos[e]["scene_name"]][infos[e]["episode_id"]]["question_token"] = infos[e]["question_token"].tolist()
                        save_data[infos[e]["scene_name"]][infos[e]["episode_id"]]["llm_object_id"] = int(infos[e]["llm_object_id"])
                        finished_episode_ids.append(infos[e]["episode_id"])
                else:
                    episode_success.append(success)
                    episode_spl.append(spl)
                    episode_dist.append(dist)
                wait_env[e] = 1.
                goal_confidence[e]=False                
                update_intrinsic_rew(e)
                init_map_and_pose_for_env(e)
        if UPDATED:
            log = "INFO: log file is updated"
            print(log)
            logging.info(log)
            with open(log_filename, "w") as f:
                json.dump(save_data,f, indent=4)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
             in range(num_scenes)])
        ).float().to(device)
        poses[:,0] = - poses[:, 0]
        # print(f"sem map input obs shape: {obs.shape}")
        _, local_map, _, local_pose = \
            sem_map_module(obs, poses, local_map, local_pose)

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        
        itm_done = []
        for e in range(num_scenes):
            itm_done.append(False)
        itm_scores = np.zeros(num_scenes)
        pred_answers = [0 for i in range(num_scenes)]
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Global Policy
        if l_step == args.num_local_steps - 1:
            obs_for_itm = envs.get_obs()
            # For every global step, update the full and local maps
            for e in range(num_scenes):
                new_epi = False
                if wait_env[e] == 1:  # New episode
                    new_epi=True
                    wait_env[e] = 0.
                else:
                    update_intrinsic_rew(e)
                
                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                    local_map[e]
                cn = int(llm_objects[e] + 4)
                # BLIP2でtext-image matching
                if (infos[e]["observe_action"]==True) and (new_epi==False):
                    

                    caption = infos[e]["llm_declarative_text"]
                    question = infos[e]["question_text"]

                    # Preprocess Image for Image Text Matching
                    if args.use_pano_for_itm==True:
                        raw_img = obs_for_itm[e]["rgb_equirectangular"]
                    else:
                        raw_img = obs_for_itm[e]["rgb_pinhole"]
                    pil_img = Image.fromarray(raw_img)
                    img = vis_processors["eval"](pil_img).unsqueeze(0).to(device)
                    # Image Text Matching
                    txt = text_processors["eval"](caption)
                    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
                    itm_output = torch.nn.functional.softmax(itm_output, dim=1)
                    text_img_prob = itm_output[:, 1].item()
                    itm_scores[e] = text_img_prob

                    # Preprocess Image for Visual Question Answering
                    if args.use_pano_for_vqa==True:
                        raw_img = obs_for_itm[e]["rgb_equirectangular"]
                    else:
                        raw_img = obs_for_itm[e]["rgb_pinhole"]
                    pil_img = Image.fromarray(raw_img)
                    img = vis_processors_vqa["eval"](pil_img).unsqueeze(0).to(device)
                    txt = text_processors_vqa["eval"](question)
                    vqa_output = model_vqa.predict_answers({"image": img, "text_input": txt}, answer_list=answer_candidates, inference_method="rank")
                    pred_answers[e] = vqa_output
                    
                    if text_img_prob <= args.text_img_matching_threshold:# SemanticMap用のcategory maskの更新
                        llm_category_masks[e] = (full_map[e,cn].to('cpu').detach().numpy().copy())
                        llm_category_masks[e] = skimage.morphology.binary_dilation(
                            llm_category_masks[e], selem) != True # objectが観測されたところは0に. 観測されていないところは1に.
                        llm_category_masks[e] = torch.from_numpy(llm_category_masks[e].astype(np.float32)).clone()
                    else:
                        goal_confidence[e] = True
                    itm_done[e]=True
                    print(e, text_img_prob, llm_category_masks[e].sum()/llm_category_masks[e].shape[0]/llm_category_masks[e].shape[1], llm_category_masks[e].shape)


                
                full_pose[e] = local_pose[e] + \
                    torch.from_numpy(origins[e]).to(device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                                        lmb[e, 0]:lmb[e, 1],
                                        lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()
            
                # Category mask
                tmp = local_map[e,cn].sum()
                local_map[e,cn] = torch.logical_and(llm_category_masks[e][lmb[e, 0]:lmb[e, 1],lmb[e, 2]:lmb[e, 3]].to(device), local_map[e,cn])
                print(f"thread {e}---- before: {tmp}, after: {local_map[e,cn].sum()}")

            locs = local_pose.cpu().numpy()
            for e in range(num_scenes):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :]
            global_input[:, 4:8, :, :] = \
                nn.MaxPool2d(args.global_downscaling)(
                    full_map[:, 0:4, :, :])
            global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()

            extras[:, 0] = global_orientation[:, 0]
            extras[:, 1] = llm_objects

            # Get exploration reward and metrics
            g_reward = torch.from_numpy(np.asarray(
                [infos[env_idx]['g_reward'] for env_idx in range(num_scenes)])
            ).float().to(device)
            g_reward += args.intrinsic_rew_coeff * intrinsic_rews.detach()


            g_process_rewards += g_reward.cpu().numpy()
            g_total_rewards = g_process_rewards * \
                (1 - g_masks.cpu().numpy())
            g_process_rewards *= g_masks.cpu().numpy()
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

            if np.sum(g_total_rewards) != 0:
                for total_rew in g_total_rewards:
                    if total_rew != 0:
                        g_episode_rewards.append(total_rew)

            # Add samples to global policy storage
            if step == 0:
                g_rollouts.obs[0].copy_(global_input)
                g_rollouts.extras[0].copy_(extras)
            else:
                # Compute additional inputs if needed
                extra_maps = {
                    "dmap": None,
                    "umap": None,
                    "fmap": None,
                    "pfs": None,
                    "agent_locations": None,
                    "ego_agent_poses": None,
                }

                extra_maps["agent_locations"] = []
                for e in range(num_scenes):
                    pose_pred = planner_pose_inputs[e]
                    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
                    gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                    map_r, map_c = start_y, start_x
                    map_loc = [
                        int(map_r * 100.0 / args.map_resolution - gx1),
                        int(map_c * 100.0 / args.map_resolution - gy1),
                    ]
                    map_loc = pu.threshold_poses(map_loc, global_input[e].shape[1:])
                    extra_maps["agent_locations"].append(map_loc)

                # get frontier map
                planner_inputs = [{} for e in range(num_scenes)]
                for e, p_input in enumerate(planner_inputs):
                    obs_map = local_map[e, 0, :, :].cpu().numpy()
                    exp_map = local_map[e, 1, :, :].cpu().numpy()
                    p_input["obs_map"] = obs_map
                    p_input["exp_map"] = exp_map
                    p_input["pose_pred"] = planner_pose_inputs[e]
                masks = [1 for _ in range(num_scenes)]
                # fmap = nf_planner.get_frontier_maps(planner_inputs, masks)
                fmap = envs.get_frontier_map(planner_inputs)
                # import pdb;pdb.pdb.set_trace()
                extra_maps["fmap"] = torch.from_numpy(fmap).to(device)

                if g_policy.needs_egocentric_transform:
                    ego_agent_poses = []
                    for e in range(num_scenes):
                        map_loc = extra_maps["agent_locations"][e]
                        # Crop map about a center
                        ego_agent_poses.append(
                            [map_loc[0], map_loc[1], math.radians(start_o)]
                        )
                    ego_agent_poses = torch.Tensor(ego_agent_poses).to(device)
                    extra_maps["ego_agent_poses"] = ego_agent_poses

                ####################################################################
                # Sample long-term goal from global policy
                _, g_action, _, _ = g_policy.act(
                    global_input,
                    None,
                    g_masks,
                    extras=extras.long(),
                    deterministic=False,
                    extra_maps=extra_maps,
                )
                
                cpu_actions = g_action.cpu().numpy()
                if len(cpu_actions.shape) == 4:
                    # Output action map
                    global_goals = cpu_actions[:, 0]  # (B, H, W)
                elif len(cpu_actions.shape) == 3:
                    # Output action map
                    global_goals = cpu_actions  # (B, H, W)
                g_masks.fill_(1.0)
                ########################################################################
                # Define long-term goal map
                ########################################################################
                goal_maps = [
                    np.zeros((local_w, local_h)) for _ in range(num_scenes)
                ]
                # Set goal to sampled location
                for e in range(num_scenes):
                    if type(global_goals) == type([]):
                        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
                    else:
                        assert len(global_goals.shape) == 3
                        goal_maps[e][:, :] = global_goals[e]
            

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------

        for e in range(num_scenes):
            cn = int(llm_objects[e] + 4)
            # print(f"Sums: {local_map[e, cn].sum()}, { (torch.logical_and(llm_category_masks[e][lmb[e, 0]:lmb[e, 1],lmb[e, 2]:lmb[e, 3]].to(device), local_map[e,cn])).sum()}")
            local_map[e,cn] = torch.logical_and(llm_category_masks[e][lmb[e, 0]:lmb[e, 1],lmb[e, 2]:lmb[e, 3]].to(device), local_map[e,cn])

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = [0 for _ in range(num_scenes)]

        for e in range(num_scenes):
            cn = infos[e]['llm_object_id'] + 4
            if local_map[e, cn, :, :].sum() >= 1.:
                print(f"Thread {e} found goal (ch.{cn}), sum: {local_map[e, cn, :, :].sum()}")
                cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                goal_maps[e] = cat_semantic_scores
                found_goal[e] = 1
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = goal_maps[e]  # global_goals[e]
            p_input['new_goal'] = l_step == args.num_local_steps - 1
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = wait_env[e] or finished[e]
            p_input["goal_confident"] = goal_confidence[e]
            p_input["itm_done"] = itm_done[e]
            p_input["itm_score"] = itm_scores[e]
            p_input["pred_answer"] = pred_answers[e]
            p_input["fmap"] = fmap[e] # to save map as png
            if args.visualize or args.print_images:
                tmp_local_map = local_map[e].detach().clone()
                tmp_local_map[-1,:,:] = 1e-5
                p_input['sem_map_pred'] = tmp_local_map[4:, :,
                                                    :].argmax(0).cpu().numpy()

        obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------

        for env_idx in range(num_scenes):
            if env_idx==0:
                llm_objects = infos[env_idx]["llm_object_id"].unsqueeze(0)
            else:
                llm_objects = torch.cat((llm_objects, infos[env_idx]["llm_object_id"].unsqueeze(0)), 0)
            
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------

    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")
        
        total_success = []
        total_spl = []
        total_dist = []
        for e in range(args.num_processes):
            for acc in episode_success[e]:
                total_success.append(acc)
            for dist in episode_dist[e]:
                total_dist.append(dist)
            for spl in episode_spl[e]:
                total_spl.append(spl)

        if len(total_spl) > 0:
            log = "Final ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                np.mean(total_success),
                np.mean(total_spl),
                np.mean(total_dist),
                len(total_spl))

        print(log)
        logging.info(log)
            
        # Save the spl per category
        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(key,
                                          sum(success_per_category[key]) /
                                          len(success_per_category[key]),
                                          sum(spl_per_category[key]) /
                                          len(spl_per_category[key]))

        print(log)
        logging.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)


if __name__ == "__main__":
    main()
