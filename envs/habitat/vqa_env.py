import json
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import quaternion
import skimage.morphology
import habitat

from envs.utils.fmm_planner import FMMPlanner
from constants import coco_categories, mp3d_categories, restrictedmp3d2idx,restrictedidx2mp3d, mp3d_categories_list_in_coco
import envs.utils.pose as pu
import h5py
from envs.utils.hab_utils import get_floor_heights

from habitat.utils.visualizations import maps
from habitat_sim.utils import common as utils
import habitat
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
import magnum as mn
import math
import re
import torch
import cv2
import logging

RGB_KEY = "rgb"
DEPTH_KEY = "depth"
SEMANTIC_KEY = "semantic"
# RGB_KEY = "rgb_resized"
# DEPTH_KEY = "depth_resized"
# SEMANTIC_KEY = "SEMANTIC_KEY"

class VQA_Env(habitat.RLEnv):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank

        super().__init__(config_env, dataset)

        # Loading dataset info file
        self.split = config_env.DATASET.SPLIT
        self.episodes_file = config_env.DATASET.EPISODES_DIR.format(
            split=self.split)
        self.dataset_info={}
        self.vertices_info = {}
        self.per_floor_dims = {}
        # Specifying action and observation space
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                 args.frame_width),
                                                dtype='uint8')

        # Initializations
        self.episode_no = -1

        # Scene info
        self.last_scene_path = None
        self.scene_path = None
        self.scene_name = None

        # Episode Dataset info
        self.eps_data = None
        self.eps_data_idx = None
        self.gt_planner = None
        self.object_boundary = None
        self.goal_idx = None
        self.goal_name = None
        self.map_obj_origin = None
        self.starting_loc = None
        self.starting_distance = None

        # Episode tracking info
        self.curr_distance = None
        self.prev_distance = None
        self.timestep = None
        self.stopped = None
        self.path_length = None
        self.last_sim_location = None
        self.trajectory_states = []
        self.info = {}
        self.info['distance_to_goal'] = None
        self.info['spl'] = None
        self.info['success'] = None
        self.scene = config_env.SIMULATOR.SCENE
        self.scene_id = "/".join(config_env.SIMULATOR.SCENE.split("/")[2:])

        self.eps_data, self.question_vocab, self.answer_vocab = self.load_eps_file_for_new_scene(self.scene_id)
        self.eps_data_idx = 0

        self.next_scene_offset = -1
        self.scenes = config_env.DATASET.CONTENT_SCENES
        self.config_env = config_env
        self.scene_name = config_env.SIMULATOR.SCENE.split("/")[-1].split(".")[0]
        self.counter = -1
        self.info["thread_finished"]=False
        self.finished = False
        self.max_eps_no_at_the_scene = len(self.eps_data)
    
    def switch_scene(self):
        scene_path_tmp = self._env._config.SIMULATOR.SCENE.split("/")[:-2]
        scene_path_tmp = "/".join(scene_path_tmp)
        self._env._reset_stats()
        self.next_scene_offset += 1
        if len(self.config_env.DATASET.CONTENT_SCENES) <=self.next_scene_offset:
            self.finished = True
            next_scene_name = self.config_env.DATASET.CONTENT_SCENES[0]
        else:
            next_scene_name = self.config_env.DATASET.CONTENT_SCENES[self.next_scene_offset]
            if next_scene_name in ["scene0569_00"]:
                self.next_scene_offset += 1
                self.next_scene_offset = self.next_scene_offset % len(self.config_env.DATASET.CONTENT_SCENES)
                next_scene_name = self.config_env.DATASET.CONTENT_SCENES[self.next_scene_offset]
        self.scene_name = next_scene_name
        if self.args.dataset == "scannet":
            next_scene = scene_path_tmp + "/" + next_scene_name + "/" + next_scene_name + "_filled" + ".glb"
        elif self.args.dataset == "mp3d":
            next_scene = scene_path_tmp + "/" + next_scene_name + "/" + next_scene_name + ".glb"
        self.scene_id = f"{self.args.dataset}"+ "/" + next_scene_name + "/" + next_scene_name + ".glb"
        print(f"Change scene to {next_scene}. scenes are {self.config_env.DATASET.CONTENT_SCENES}")
        self._env._config.defrost()
        self._env._config.SIMULATOR.SCENE = next_scene
        self._env._config.freeze()
        self._env._sim.reconfigure(self._env._config.SIMULATOR)

        self._env.current_episode = next(self._env._episode_iterator)
        observations = self._env.task.reset(episode=self._env.current_episode)
        self._env._task.measurements.reset_measures(
            episode=self._env.current_episode,
            task=self._env.task,
            observations=observations,
        )
        self.eps_data, _, _ = self.load_eps_file_for_new_scene(self.scene_id)
        self.eps_data_idx = 0
        self.max_eps_no_at_the_scene = len(self.eps_data)
        return observations

    def load_eps_file_for_new_scene(self, scene_id):
        eps_data = []
        with gzip.open(self.episodes_file, 'r') as f:
            json_data = json.loads(
                f.read().decode('utf-8'))
        question_vocab =  json_data['question_vocab']
        answer_vocab =  json_data['answer_vocab']
        for episode in json_data["episodes"]:
            if self.args.dataset=="scannet":
                if episode["scene_id"].split("/")[0] == scene_id.split("/")[1]:
                    eps_data.append(episode)
            elif self.args.dataset=="mp3d":
                if episode["scene_id"].split("/")[1] == scene_id.split("/")[1]:
                    eps_data.append(episode)
        return eps_data, question_vocab, answer_vocab
    
    def load_new_episode_vqa(self):
        args = self.args
        self.scene_path = self._env._config.SIMULATOR.SCENE
        scene_name = self.scene_name
        # log = f"current offset: {self.eps_data_idx}, max offset: {len(self.eps_data)}, {self.max_eps_no_at_the_scene}"
        # logging.info(log)
        # print(log)
        episode = self.eps_data[self.eps_data_idx]
        self.episode = episode
        self.eps_data_idx += 1
        self.gt_eps_idx = episode["episode_id"]
        goal_name = episode["info"]["bboxes"][0]["name"]
        goal_instance_idx = episode["info"]["bboxes"][0]["box"]["obj_id"]
        question_token = torch.tensor(episode["question"]["question_tokens"])
        question_text = episode["question"]["question_text"]
        self.answer_id = goal_instance_idx
        self.question_token = torch.cat((question_token, torch.zeros(30 - question_token.shape[0])), axis=0)
        self.question_text = question_text
        self.answer_text = episode["question"]['answer_text']

        if args.back_step_num=="rand_init":
            start_offset=0
        elif self.args.back_step_num in ["10", "30", "50"]:
            start_offset = len(self.episode['shortest_paths'][0]) -1 - int(self.args.back_step_num)
            if start_offset <= -1:
                start_offset=0
        elif self.args.back_step_num in ["0"]:
            start_offset = -1
        else:
            raise Exception
        pos = episode['shortest_paths'][0][start_offset]["position"]
        rot = episode['shortest_paths'][0][start_offset]["rotation"]
        if self.args.dataset=="scannet":
            rot = np.quaternion(rot[0], rot[1], rot[2], rot[3])
        self.goal_object_idx = goal_instance_idx
        self.goal_name = goal_name
        self._env.sim.set_agent_state(pos, rot)
        obs = self._env.sim.get_observations_at(pos, rot)
        obs = self._env.step(action={"action": 2})
        obs = self._env.step(action={"action": 3})
        return obs

    def reset_init(self):
        self.eps_data_idx = 0
    
    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        args = self.args
        self.counter +=1
        if (self.counter % args.num_train_episodes == 0) or (self.eps_data_idx >= self.max_eps_no_at_the_scene) :
            change_scene = True
        else:
            change_scene = False
        if self.finished:
            self.info["thread_finished"]=True


        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []
        # if (new_scene) or (self.episode_no>=2):
        if self.info["thread_finished"]==False:
            if change_scene:
                obs = self.switch_scene()
                self.reset_init()
            self.scene_path = self.scene_name
            obs = self.load_new_episode_vqa()
        else:
            state = np.zeros(self.__state_shape__)
            return state, self.info
        # print(len(self.config_env.DATASET.CONTENT_SCENES), self.next_scene_offset, self.finished, self.episode_no)

        # rgb = obs['rgb'].astype(np.uint8)
        rgb = obs[RGB_KEY].astype(np.uint8)
        depth = obs[DEPTH_KEY]
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.__state_shape__ = state.shape
        self.last_sim_location = self.get_sim_location()
        self._obs = obs

        # Set info
        self.info["episode_id"]=self.episode["episode_id"]
        self.info["answer"] = self.episode["question"]["answer_text"]
        self.info['question_type'] = self.episode["question"]['question_type']
        self.info["start_position"]=self.episode["start_position"]
        self.info["scene_name"] = self.scene_name
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['goal_name'] = self.goal_name
        self.info["question_token"] = self.question_token
        self.info["question_text"] = self.question_text
        # self.info["llm_declarative_text"] = self.declarative_text
        # self.info["distance_to_goal"] = self.starting_distance - self.object_boundary
        self.info["observe_action"] = 0
        self.info["sem_contain_goal"] = self.if_sem_contain_goal(obs["semantic_equirectangular"], self.answer_id)


        return state, self.info

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        # action = action["action"]
        action["answer_id"] = self.answer_id
        spl, success, dist = 0., 0., 0.
        done=True

        return 0, 0, done, self.info
    
    def get_obs(self):
        obs = {}
        obs["rgb_equirectangular"] = cv2.cvtColor(self._obs["rgb_equirectangular"], cv2.COLOR_BGR2RGB)
        obs["rgb_pinhole"] = cv2.cvtColor(self._obs[RGB_KEY], cv2.COLOR_BGR2RGB)
        obs["semantic_equirectangular"] = self._obs["semantic_equirectangular"]
        obs["semantic_pinhole"] = self._obs[SEMANTIC_KEY]
        # obs[""]
        return obs
    
    def if_sem_contain_goal(self, sem_array, goal_id):
        return goal_id in np.unique(sem_array)

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)

    def get_done(self, observations):
        if self.info['time'] >= self.args.max_episode_length - 1:
            done = True
        elif self.stopped:
            done = True
        else:
            done = False
        return done

    def get_info(self, observations):
        """This function is not used, Habitat-RLEnv requires this function"""
        info = {}
        return info

    def get_spaces(self):
        """Returns observation and action spaces for the ObjectGoal task."""
        return self.observation_space, self.action_space

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = agent_state.position[2]
        y = agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                        (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
                
        return x, y, o

