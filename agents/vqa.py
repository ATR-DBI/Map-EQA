import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
from torchvision import transforms

from envs.utils.fmm_planner import FMMPlanner
from envs.habitat.vqa_env import VQA_Env
from agents.utils.semantic_prediction import SemanticPredMaskRCNN, SemanticPredMaskRCNN_mp3d
import envs.utils.pose as pu
import agents.utils.visualization as vu
import torch


class VQA_Env_Agent(VQA_Env): 
    """
    This env anget gathers images at a designated backstep positions
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

        self.sem_pred = SemanticPredMaskRCNN_mp3d(args)

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)

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
            self.legend = cv2.imread('docs/legend.png')
            if self.args.num_sem_categories==21:
                self.legend = cv2.imread('docs/legend21.jpg')
            self.vis_image = None
            self.rgb_vis = None

    def reset(self):
        args = self.args

        obs, info = super().reset()
        # obs = self._preprocess_obs(obs)

        self.obs_shape = obs.shape

        # Episode initializations
        self.last_action = None
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
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            obs = self._obs
            return 0, 0, True, self.info

        action=3
        # if self.args.visualize or self.args.print_images:
        #     self._visualize(planner_inputs)
        #     if planner_inputs["took_panorama"]==True:
        #         self._visualize_pano()

        # act
        action = {'action': action}
        obs, rew, done, info = super().step(action)
        if info["thread_finished"]==False:
            self._save_image()
        else:
            pass

        self.last_action = action['action']
        self.info = info
        done=True
        return np.array([0]), rew, done, info

    def _visualize_pano(self):
        args = self.args
        apend = self.args.back_step_num
        if self.args.back_step_num in ["10", "30", "50"]:
            apend = f"T_{self.args.back_step_num}"
        elif self.args.back_step_num in ["0"]:
            apend = "end"
        im_dir = f"vqa/{self.args.dataset}/images/{apend}/{self.args.split}"
        
        obs = self.get_obs()
        pano = obs["rgb_equirectangular"]
        rgb = cv2.cvtColor(np.asarray(obs["rgb"]), cv2.COLOR_BGR2RGB)
        if args.print_images:
            fn = im_dir + '/pano/{}.png'.format(self.gt_eps_idx)
            cv2.imwrite(fn, pano)
            fn = im_dir + '/pinhole/{}.png'.format(self.gt_eps_idx)
            cv2.imwrite(fn, rgb)
    
    def _save_image(self):
        apend = self.args.back_step_num
        if self.args.back_step_num in ["10", "30", "50"]:
            apend = f"T_{self.args.back_step_num}"
        elif (self.args.back_step_num in ["0", "end"]):
            apend = "end"
        save_dir = f"vqa/{self.args.dataset}/images/{apend}/{self.args.dataset_version}/{self.args.split}/pinhole"
        fn = f"{save_dir}/{str(self.gt_eps_idx)}.png"
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                pass
        img = self.get_obs()["rgb_pinhole"]
        if not os.path.exists(fn):
            cv2.imwrite(fn, img)
        else:
            pass
        
        return None
    
    def _save_depth(self):
        apend = self.args.back_step_num
        if self.args.back_step_num in ["10", "30", "50"]:
            apend = f"T_{self.args.back_step_num}"
        elif self.args.back_step_num in ["0"]:
            apend = "end"
        depth_dir = f"vqa/{self.args.dataset}/depth/{apend}/{self.args.split}/pinhole"
        if not os.path.exists(depth_dir):
            try:
                os.makedirs(depth_dir)
            except:
                pass
        obs = self.get_obs()
        depth = obs["depth"]
        fn = depth_dir + '/{}.npy'.format(self.gt_eps_idx)
        np.save(fn, depth)
    
    def _save_semantic(self):
        apend = self.args.back_step_num
        if self.args.back_step_num in ["10", "30", "50"]:
            apend = f"T_{self.args.back_step_num}"
        elif self.args.back_step_num in ["0"]:
            apend = "end"
        sem_dir = f"vqa/{self.args.dataset}/semantics/{apend}/{self.args.dataset_version}/{self.args.split}/pinhole"
        if not os.path.exists(sem_dir):
            try:
                os.makedirs(sem_dir)
            except:
                pass
        obs = self.get_obs()
        # sem_pano = obs["semantic_equirectangular"]
        sem = obs["semantic"]
        # fn = sem_dir + '/pano/{}.npy'.format(self.gt_eps_idx)
        # np.save(fn, sem_pano)
        fn = sem_dir + '/{}.npy'.format(self.gt_eps_idx)
        if not os.path.exists(fn):
            np.save(fn, sem)
        else:
            pass
 