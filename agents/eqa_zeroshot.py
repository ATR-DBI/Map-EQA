import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
from torchvision import transforms

from envs.utils.fmm_planner import FMMPlanner
from envs.habitat.eqa_zeroshot_env import EQA_Zeroshot_Env
from agents.utils.detic_prediction import SemanticPredDetic
from constants import color_palette, rednet_color_palette
import envs.utils.pose as pu
import agents.utils.visualization as vu
import torch

# For frontier exploration
from agents.policy.nf_planner import PlannerActor
import libs.astar_pycpp.pyastar as pyastar
import time

def visualize_sem_map(sem_map):
    c_map = sem_map.astype(np.int32)
    color_palette = rednet_color_palette
    semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
    semantic_img.putpalette(color_palette)
    semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = np.array(semantic_img)
    return semantic_img

class EQA_Zeroshot_Env_Agent(EQA_Zeroshot_Env):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        super().__init__(args, rank, config_env, dataset)

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        if self.args.sem_model=="detic":
            self.sem_pred = SemanticPredDetic(args)
            

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)
        self.stg_selem = skimage.morphology.disk(self.args.stg_disk_size)
        self.stg_selem_close = skimage.morphology.disk(self.args.stg_disk_size // 2)
        self.obs_selem = skimage.morphology.disk(self.args.obs_disk_size)
        self.semseg_selem = skimage.morphology.disk(self.args.semseg_shrink_disk_size)

        self.planner = PlannerActor(args)
        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None

        if args.visualize or args.print_images:
            # self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None

        self.time_start = time.time()
        self.time_end = time.time()
        self.time_end_reset = time.time()

    def reset(self):
        args = self.args
        self.time_start = time.time()

        obs, info = super().reset()
        if self.args.sem_model=="detic":
            new_vocab = [self.llm_object]
            self.sem_pred.model.reset_vocab(new_vocab=new_vocab, vocab_type="custom")
        obs = self._preprocess_obs(obs)
        

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self.panos = []
        self.itm_vqa_infos = {
            "itm_score": [],
            "pred_answer": None,
            "obs_itm_vqa": [],
            "time": []
        }
        if args.visualize or args.print_images:
            # self.vis_image = vu.eqa_init_vis_image(self.question_text, self.legend)
            self.vis_image = vu.eqa_init_vis_image(self.question_text)

        return obs, info

    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return (np.zeros(self.obs_shape)).astype(np.float32), 0., False, self.info

        # Reset reward if new long-term goal
        if planner_inputs["new_goal"]:
            self.info["g_reward"] = 0

        # get action
        action = self._plan(planner_inputs)

        if (self.args.visualize or self.args.print_images) and (self.info["thread_finished"]==False):
            self._visualize(planner_inputs)


        # act
        if self.info["time"] >= self.args.max_episode_length:
            action={'action': 0}
            done=True
        else:
            action = {'action': action}
        obs, rew, _, info = super().step(action)
        

        # preprocess obs
        obs = self._preprocess_obs(obs) 
        self.last_action = action['action']
        self.obs = obs
        self.obs_shape = obs.shape # in the case when this thread is finished but has to send any data like zero_array.
        
        
        info["observe_action"]=self.info["observe_action"]
        info["observe_finished"]=self.info["observe_finished"]
        self.info = info


        info['g_reward'] += rew
        done = info["observe_finished"]
        if info["time"] > self.args.max_episode_length:
            done=True

        self.time_end = time.time()
        if done:
            spl, success, dist = 0,0,0
            d_T, d_D, d_min, d_0 = self.get_metrics_dtt()
            info["in_gt_semmap"]=0
            spl, success, dist = self.get_metrics()
            info["spl_without_succ_fail"] = min(self.dists_to_target[0]/ self.path_len, 1)
            info['distance_to_goal'] = dist
            info['spl'] = spl
            info['success'] = success
            info["pred_answer"] = self.itm_vqa_infos["pred_answer"]
            info["itm_score"] = self.itm_vqa_infos["itm_score"]
            info["itm_time"] = self.itm_vqa_infos["time"]
            info["vqa_success"] = info["pred_answer"][0]==self.answer_text
            info["pacman"]["d_T"] = d_T
            info["pacman"]["d_D"] = d_D
            info["pacman"]["d_min"] = d_min
            info["pacman"]["d_0"] = d_0
            info["time_per_episode"] = self.time_end - self.time_start
            self._save_itm_images()
            # if self.args.save_all_itm_images:
            #     self._save_itm_images()
            # else:
            #     self._save_image()
        return obs, rew, done, info

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found
                    "goal_confident": (bool)

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        if args.visualize or args.print_images:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0 / args.map_resolution - gx1),
                          int(c * 100.0 / args.map_resolution - gy1)]
            last_start = pu.threshold_poses(last_start, map_pred.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1
        if planner_inputs["found_goal"]:
            is_agent_within_a_dist_to_target=self._check_within_dist(map_pred, start, np.copy(goal),
                                    planning_window) # true if the agent is within a certain distance to a target object
        else:
            is_agent_within_a_dist_to_target=False
        
        
        if planner_inputs["itm_done"]==True:
            self.stock_itm_vqa_infos(itm_score=planner_inputs["itm_score"], pred_answer=planner_inputs["pred_answer"])
            if planner_inputs["goal_confident"]==True:
                self.info["observe_finished"] = True
            else:
                self.info["observe_finished"] = False
        else:
            self.info["observe_finished"] = False


        stg, replan, stop = self._get_stg_frontier(map_pred, start, np.copy(goal),
                                planning_window)

        # when the agent is within a certain distance to a target object, the agent looks towards the object
        if is_agent_within_a_dist_to_target: # if true, stgをtarget object座標のmeanとなるように設定
            indices = np.where(planner_inputs['goal']==1)
            stg_x = np.mean(indices[0])
            stg_y = np.mean(indices[1])
            self.stg = (stg_x, stg_y)
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action=0 # stop
            # for ITM
            if action==0:
                do_observe=True
            else:
                do_observe=False
            stop = do_observe
            
        else:
            do_observe=False
        
        self.do_observe = do_observe
        if self.info['time'] >= self.args.max_episode_length  - 2:
            do_observe=True
            self.do_observe = True
        self.info["observe_action"]=self.do_observe
        
        # Deterministic Local Policy
        if stop:
            action = 0  # Stop
        elif is_agent_within_a_dist_to_target:
            pass
        else:
            (stg_x, stg_y) = stg
            self.stg = (stg_x, stg_y)
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward # TODO: from this code, the agent might collide with objects

        return action

    def _check_within_dist(self, grid, start, goal, planning_window): # target objectに一定以上近づいている場合にはTrueを返す
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(int((self.args.stg_stop_dist)*100. / self.args.map_resolution)) # success_dist + 0.5 mの距離でstop
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)
        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        _, _, _, stop = planner.get_short_term_goal(state)
        return stop
    
    # stock itm score and timestamp for 
    def stock_itm_vqa_infos(self, itm_score, pred_answer):
        self.itm_vqa_infos["itm_score"].append(itm_score)
        self.itm_vqa_infos["time"].append(self.info["time"])
        self.itm_vqa_infos["pred_answer"] = pred_answer
        self.itm_vqa_infos["obs_itm_vqa"].append(self.get_obs())


    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop
    
    def _get_stg_frontier(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        s = self.args.stg_downsampling
        traversible[
            int(start[0] - x1) - s : int(start[0] - x1) + s + 1,
            int(start[1] - y1) - s : int(start[1] - y1) + s + 1,
        ] = 1

        traversible = self.add_boundary(traversible)
        goal = self.add_boundary(goal, value=0)

        scale = self.args.stg_downsampling
        step_size = 5 if scale == 1 else 3 * scale
        selem = self.stg_selem
        goal = cv2.dilate(goal, selem)

        if True:
            stg_x, stg_y = None, None
            obstacles = (1 - traversible).astype(np.float32)
            astar_goal = goal.astype(np.float32)
            astar_start = [int(start[1] - y1 + 1), int(start[0] - x1 + 1)]
            if scale != 1:
                obstacles = obstacles[::scale]
                astar_goal = astar_goal[::scale]
                astar_start = [astar_start[0] // scale, astar_start[1] // scale]
                step_size = step_size // scale
            path_y, path_x = pyastar.multi_goal_weighted_astar_planner(
                obstacles,
                astar_start,
                astar_goal,
                True,
                wscale=self.args.weighted_scale,
                niters=self.args.weighted_niters,
            )
            if scale != 1:
                path_y = [y * scale for y in path_y]
                path_x = [x * scale for x in path_x]
            if path_x is not None:
                # The paths are in reversed order
                stg_x = path_x[-min(step_size, len(path_x))]
                stg_y = path_y[-min(step_size, len(path_y))]
                replan = False
                stop = False
                if len(path_x) < step_size:
                    # Measure distance along the shortest path
                    path_xy = np.stack([path_x, path_y], axis=1)
                    d2g = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1)
                    d2g = d2g.sum() * self.args.map_resolution / 100.0  # In meters
                    if d2g <= 0.25:
                        stop = True

            if stg_x is None:
                # Pick some arbitrary location as the short-term goal
                random_theta = np.random.uniform(-np.pi, np.pi, (1,))[0].item()
                stg_x = int(step_size * np.cos(random_theta))
                stg_y = int(step_size * np.sin(random_theta))
                replan = True
                stop = False

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), replan, stop

    def add_boundary(self, mat, value=1):
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2)) + value
        new_mat[1 : h + 1, 1 : w + 1] = mat
        return new_mat

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        
        if self.args.sem_model=="detic":
            sem_seg_pred = self._get_sem_pred_detic(rgb.astype(np.uint8), depth.copy())
            depth = self._preprocess_depth_wonormalize(depth, args.min_depth, args.max_depth)


        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]): # replace 0 to the max value of the observed depth
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth
    
    def _preprocess_depth_wonormalize(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1
        depth[depth <= 0.] = max_d

        mask2 = depth > 0.99*max_d
        depth[mask2] = 0.

        mask1 = depth <= min_d
        depth[mask1] = 100.0*max_d
        depth = depth * 100.0
        return depth

    def _get_sem_pred_detic(self, rgb, depth, use_seg=True):
        depth = torch.from_numpy(depth).to(f"cuda:{self.args.sem_gpu_id}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[np.newaxis,...]
        self.rgb_vis = rgb[0]
        rgb = np.transpose(rgb, (0,3,1,2))
        rgb = torch.from_numpy(rgb).to(f"cuda:{self.args.sem_gpu_id}")
        _semantic_pred = self.sem_pred.get_predictions(rgb,depth)
        semantic_pred = (skimage.morphology.binary_erosion(_semantic_pred.cpu()[0,0], self.semseg_selem)*1)
        semantic_pred = semantic_pred[np.newaxis,...].copy()[np.newaxis,...]
        
        semantic_pred = np.transpose(semantic_pred, (0,2,3,1))
        semantic_pred = semantic_pred.astype(np.float32)[0]

        _semantic_pred = _semantic_pred.to('cpu').detach().numpy().copy()
        _semantic_pred = np.transpose(_semantic_pred, (0,2,3,1))
        _semantic_pred = _semantic_pred.astype(np.float32)[0]

        self.sem_seg_pred = semantic_pred
        self.sem_seg_pred_before_shrink = _semantic_pred
        return semantic_pred
    
    def get_frontier_map(self, planner_inputs):
        """Function responsible for computing frontiers in the input map

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'obs_map' (ndarray): (M, M) map of obstacle locations
                    'exp_map' (ndarray): (M, M) map of explored locations

        Returns:
            frontier_map (ndarray): (M, M) binary map of frontier locations
        """
        cfg = self.args

        obs_map = np.rint(planner_inputs["obs_map"])
        exp_map = np.rint(planner_inputs["exp_map"])
        # compute free and unexplored maps
        free_map = (1 - obs_map) * exp_map
        unk_map = 1 - exp_map
        # Clean maps
        kernel = np.ones((5, 5), dtype=np.uint8)
        free_map = cv2.morphologyEx(free_map, cv2.MORPH_CLOSE, kernel)
        unk_map[free_map == 1] = 0
        # https://github.com/facebookresearch/exploring_exploration/blob/09d3f9b8703162fcc0974989e60f8cd5b47d4d39/exploring_exploration/models/frontier_agent.py#L132
        unk_map_shiftup = np.pad(
            unk_map, ((0, 1), (0, 0)), mode="constant", constant_values=0
        )[1:, :]
        unk_map_shiftdown = np.pad(
            unk_map, ((1, 0), (0, 0)), mode="constant", constant_values=0
        )[:-1, :]
        unk_map_shiftleft = np.pad(
            unk_map, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )[:, 1:]
        unk_map_shiftright = np.pad(
            unk_map, ((0, 0), (1, 0)), mode="constant", constant_values=0
        )[:, :-1]
        frontiers = (
            (free_map == unk_map_shiftup)
            | (free_map == unk_map_shiftdown)
            | (free_map == unk_map_shiftleft)
            | (free_map == unk_map_shiftright)
        ) & (
            free_map == 1
        )  # (H, W)
        frontiers = frontiers.astype(np.uint8)
        # Select only large-enough frontiers
        contours, _ = cv2.findContours(
            frontiers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) > 0:
            contours = [c[:, 0].tolist() for c in contours]  # Clean format
            new_frontiers = np.zeros_like(frontiers)
            # Only pick largest 5 frontiers
            contours = sorted(contours, key=lambda x: len(x), reverse=True)
            for contour in contours[:5]:
                contour = np.array(contour)
                # Select only the central point of the contour
                lc = len(contour)
                if lc > 0:
                    new_frontiers[contour[lc // 2, 1], contour[lc // 2, 0]] = 1
            frontiers = new_frontiers
        frontiers = frontiers > 0
        # Mask out frontiers very close to the agent
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        ## Convert current location to map coordinates
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / cfg.map_resolution - gx1),
            int(c * 100.0 / cfg.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, frontiers.shape)
        ## Mask out a 100.0 x 100.0 cm region center on the agent
        ncells = int(100.0 / cfg.map_resolution)
        frontiers[
            (start[0] - ncells) : (start[0] + ncells + 1),
            (start[1] - ncells) : (start[1] + ncells + 1),
        ] = False
        # Handle edge case where frontier becomes zero
        if not np.any(frontiers):
            # Set a random location to True
            rand_y = np.random.randint(start[0] - ncells, start[0] + ncells + 1)
            rand_x = np.random.randint(start[1] - ncells, start[1] + ncells + 1)
            frontiers[rand_y, rand_x] = True

        return frontiers

    def _visualize(self, inputs, with_goal=True):
        args = self.args
        append = args.back_step_num
        if args.back_step_num in ["10", "30", "50"]:
            append = f"T_{args.back_step_num}_sem{str(self.args.detic_confidence_threshold)}_itm{str(self.args.text_img_matching_threshold)}_stop_dist_{str(args.stg_stop_dist)}"
        else:
            append = f"{args.back_step_num}_sem{str(self.args.detic_confidence_threshold)}_itm{str(self.args.text_img_matching_threshold)}_stop_dist_{str(args.stg_stop_dist)}"
        dump_dir = "{}/dump/{}/{}/{}/{}/".format(args.dump_location,
                                        args.exp_name, args.dataset, args.dataset_version,append)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.gt_eps_idx)
        if not os.path.exists(ep_dir):
            try:
                os.makedirs(ep_dir)
            except:
                pass

        map_pred = inputs['map_pred'].copy()
        exp_pred = inputs['exp_pred'].copy()
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        goal = inputs['goal'].copy()
        sem_map = inputs['sem_map_pred'].copy()

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        if self.args.num_sem_categories==8:
            no_cat_mask = sem_map == 12
        elif self.args.num_sem_categories==21:
            no_cat_mask = sem_map == 25
        elif self.args.num_sem_categories==33:
            no_cat_mask = sem_map == 37
        elif self.args.num_sem_categories==1:
            no_cat_mask = sem_map == 5
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        if with_goal:
            sem_map[goal_mask] = 4
        
        if self.args.num_sem_categories==8:
            color_pal = [int(x * 255.) for x in color_palette]
        elif self.args.num_sem_categories==21:
            color_pal = rednet_color_palette
        elif self.args.num_sem_categories==33:
            color_pal = rednet_color_palette
        elif self.args.num_sem_categories==1:
            color_pal = rednet_color_palette
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530,15:655] = self.sem_pred.rgb_with_bbox[0]
        self.vis_image[50:530, 670:1150] = sem_map_vis

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            if with_goal:
                fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                    dump_dir, self.rank, self.gt_eps_idx,
                    self.rank, self.episode_no, self.timestep)
            else:
                fn = '{}/episodes/thread_{}/eps_{}/wogoal_{}-{}-Vis-{}.png'.format(
                    dump_dir, self.rank, self.gt_eps_idx,
                    self.rank, self.episode_no, self.timestep)
            cv2.imwrite(fn, self.vis_image)


    def preprocess_depth(self, depth):
        min_depth = self.args.min_depth
        max_depth = self.args.max_depth

        # Column-wise post-processing
        depth = depth * 1.0
        H = depth.shape[1]
        depth_max, _ = depth.max(dim=1, keepdim=True)  # (B, H, W, 1)
        depth_max = depth_max.expand(-1, H, -1, -1)
        depth[depth == 0] = depth_max[depth == 0]

        mask2 = depth > 0.99
        depth[mask2] = 0.0

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0
        return depth
    
    def save_map(self, map, map_name):
        args = self.args
        dump_dir = "{}/dump/{}/T_{}/".format(args.dump_location,
                                        args.exp_name, args.back_step_num)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.gt_eps_idx)
        if not os.path.exists(ep_dir):
            try:
                os.makedirs(ep_dir)
            except:
                pass
        fn = '{}/episodes/thread_{}/eps_{}/{}_{}-{}-Vis-{}.png'.format(
            dump_dir, self.rank, self.gt_eps_idx, map_name,
            self.rank, self.gt_eps_idx, self.timestep)
        cv2.imwrite(fn, map*255)

    def _save_image(self):
        apend = self.args.back_step_num
        if self.args.back_step_num in ["10", "30", "50"]:
            apend = f"T_{self.args.back_step_num}"
        save_dir = f"vqa/{self.args.dataset}/images/after_nav/{apend}_sem{str(self.args.detic_confidence_threshold)}_itm{str(self.args.text_img_matching_threshold)}_stop_dist_{str(self.args.stg_stop_dist)}/{self.args.dataset_version}/val/pinhole"
        fn = f"{save_dir}/{str(self.gt_eps_idx)}.png"
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                pass
        img = self.get_obs()["rgb_pinhole"]
        # print(img)
        cv2.imwrite(fn, img)
        return None
    
    def _save_itm_images(self):
        apend = self.args.back_step_num
        if self.args.back_step_num in ["10", "30", "50"]:
            apend = f"T_{self.args.back_step_num}"
        save_dir = f"vqa/{self.args.dataset}/images/after_nav/{apend}_sem{str(self.args.detic_confidence_threshold)}_itm{str(self.args.text_img_matching_threshold)}_stop_dist_{str(self.args.stg_stop_dist)}/{self.args.dataset_version}/{self.args.split}/pinhole"
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                pass
        for i in range(len(self.itm_vqa_infos["time"])):
            img = self.itm_vqa_infos["obs_itm_vqa"][i]["rgb_pinhole"]
            time = self.itm_vqa_infos["time"][i]
            fn = f"{save_dir}/{str(self.gt_eps_idx)}_{time}.png"
            cv2.imwrite(fn, img)
        return None

    def _save_start_images(self):
        apend = self.args.back_step_num
        if self.args.back_step_num in ["10", "30", "50"]:
            apend = f"T_{self.args.back_step_num}"
        save_dir = f"vqa/{self.args.dataset}/images/{apend}/{self.args.dataset_version}/{self.args.split}/pinhole"
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                pass

        img = self.get_obs()["rgb_pinhole"]
        fn = f"{save_dir}/{str(self.gt_eps_idx)}.png"
        cv2.imwrite(fn, img)
        return None

    # just for debug
    # def _save_image_with_path(self, img, map_name):
    #     args = self.args
    #     apend = self.args.back_step_num
    #     if self.args.back_step_num in ["10", "30", "50"]:
    #         apend = f"T_{self.args.back_step_num}"
    #     save_dir = "tmp/eqa/episodes/thread_{}/eps_{}/".format(
    #         self.rank, self.gt_eps_idx)
    #     if not os.path.exists(save_dir):
    #         try:
    #             os.makedirs(save_dir)
    #         except:
    #             pass
    #     fn = 'tmp/eqa/episodes/thread_{}/eps_{}/{}_{}-{}-Vis-{}.png'.format(
    #         self.rank, self.gt_eps_idx, map_name,
    #         self.rank, self.gt_eps_idx, self.timestep)
    #     cv2.imwrite(fn, img)
