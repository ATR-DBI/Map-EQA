import json
import gzip
import gym
import numpy as np
import quaternion
import skimage.morphology

from envs.utils.fmm_planner import FMMPlanner
import envs.utils.pose as pu

from habitat.utils.visualizations import maps
import habitat
import re
import torch
import cv2
import time

RGB_KEY = "rgb"
DEPTH_KEY = "depth"
SEMANTIC_KEY = "semantic"
# RGB_KEY = "rgb_resized"
# DEPTH_KEY = "depth_resized"
# SEMANTIC_KEY = "semantic_resized"

# def change_map_format(semantic_map, extract_map=restrictedmp3d2idx):
#     assert len(semantic_map.shape)==2
#     category_num=len(extract_map)
#     semantic_map = np.array(semantic_map)
#     reformed_map = np.zeros((category_num+1, semantic_map.shape[0], semantic_map.shape[1])) # 最初にnavigatable areaのchannelを作るため
#     num_raw_categories = len(mp3d_categories)
#     for j in range(num_raw_categories):
#         if j ==1:# navigatable areaをとりあえずはfloorとしています
#             tmp1, tmp2 = np.where(np.array(semantic_map) == j)
#             reformed_map[0][tmp1,tmp2] = 1
#         if j in extract_map.keys():
#             i = extract_map[j]
#             tmp1, tmp2 = np.where(np.array(semantic_map) == j)
#             reformed_map[i+1][tmp1,tmp2] = 1
#     return reformed_map


class EQA_Zeroshot_Env(habitat.RLEnv):
    """The EQA environment class. The class is responsible
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
        self.episode_no = 0

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
        self.info["pacman"] = {}
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
        self.dists_to_target = []
        self.finished = False
        self.max_eps_no_at_the_scene = len(self.eps_data)

    
    def switch_scene(self):
        scene_path_tmp = self._env._config.SIMULATOR.SCENE.split("/")[:-2]
        scene_path_tmp = "/".join(scene_path_tmp)
        self._env._reset_stats()
        self.next_scene_offset += 1
        if len(self.config_env.DATASET.CONTENT_SCENES) <=self.next_scene_offset:
            if self.args.eval ==0:#train
                self.next_scene_offset = self.next_scene_offset % len(self.config_env.DATASET.CONTENT_SCENES)
                next_scene_name = self.config_env.DATASET.CONTENT_SCENES[self.next_scene_offset]
            elif self.args.eval ==1:#eval
                self.finished = True
                next_scene_name = self.config_env.DATASET.CONTENT_SCENES[0]
        else:
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

    def load_new_episode_wo_semmap(self):
        """The function loads a fixed episode from the episode dataset. This
        function is used for evaluating a trained model on the val split.
        """
        args = self.args
        self.scene_path = self._env._config.SIMULATOR.SCENE
        scene_name = self.scene_name
        # question_token_length = 15
        question_token_length = 30

        episode = self.eps_data[self.eps_data_idx]
        self.episode = episode
        self.eps_data_idx += 1
        self.eps_data_idx = self.eps_data_idx % len(self.eps_data)
        self.gt_eps_idx = episode["episode_id"]
        goal_name = episode["info"]["bboxes"][0]["name"]
        goal_instance_idx = episode["info"]["bboxes"][0]["box"]["obj_id"]
        question_token = torch.tensor(episode["question"]["question_tokens"])
        question_text = episode["question"]["question_text"]
        self.answer_id = goal_instance_idx
        self.question_token = torch.cat((question_token, torch.zeros(question_token_length - question_token.shape[0])), axis=0)
        self.question_text = question_text
        self.answer_text = episode["question"]['answer_text']
        self.llm_object = episode["llm"]["object"]
        self.goal_pos = episode['goals'][0]['position']
        
        self.declarative_text = episode["llm"]["declarative_text"]
        if self.args.back_step_num=="rand_init":
            start_offset = 0
        elif self.args.back_step_num in ["10", "30", "50"]:
            start_offset = len(self.episode['shortest_paths'][0]) -1 - int(self.args.back_step_num)
        else:
            raise Exception
        
        if start_offset < 0: # the length of shortest paths in some episodes is below 30
            start_offset = 0
        pos = episode['shortest_paths'][0][start_offset]["position"]
        rot = episode['shortest_paths'][0][start_offset]["rotation"]
        if self.args.dataset == "scannet":
            rot = np.quaternion(rot[0], rot[1], rot[2], rot[3])

        goal_idx = 0 # 決め打ち
        self.llm_object_id = torch.tensor(0)  # 決め打ち

        meters_per_pixel=0.05
        top_down_map = maps.get_topdown_map(
            self._env.sim.pathfinder, height=pos[1], meters_per_pixel=meters_per_pixel
        )
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        self.grid_dimensions = grid_dimensions
        sem_map = (top_down_map==1).astype(np.uint8)

        # Setup ground truth planner
        object_boundary = args.stg_stop_dist
        map_resolution = args.map_resolution
        selem = skimage.morphology.disk(2)
        traversible = skimage.morphology.binary_dilation(
            sem_map, selem) != True
        traversible = 1 - traversible
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(
            int(object_boundary * 100. / map_resolution))
        goal_instance_map = np.zeros(sem_map.shape)
        answer_obj_grid_pos = maps.to_grid(
            self.goal_pos[2], self.goal_pos[0], grid_dimensions, pathfinder=self._env.sim.pathfinder
        )# (y,x)
        goal_instance_map[answer_obj_grid_pos] = 1
        goal_map = skimage.morphology.binary_dilation(
            goal_instance_map, selem) != True
        goal_map = 1 - goal_map
        try:
            planner.set_multi_goal(goal_map)
        except:
            print(f'########{scene_name}, {goal_instance_idx}, {goal_name}, sum: {goal_map.sum()}')
        # Get starting loc in GT map coordinates
        self.gt_planner = planner
        self.object_boundary = object_boundary
        
        map_loc = maps.to_grid(
            pos[2], pos[0], grid_dimensions, pathfinder=self._env.sim.pathfinder
        )
        self.starting_loc = map_loc
        self.goal_object_idx = goal_instance_idx
        self.goal_name = goal_name
        self.goal_idx = goal_idx
        
        self.starting_distance = self.gt_planner.fmm_dist[self.starting_loc]\
            / 20.0 + self.object_boundary
        self.prev_distance = self.starting_distance
        self._env.sim.set_agent_state(pos, rot) 
        # i don't know why but the agent height is higher than the defined height at config. 
        # _env.step will modify the height to the correct one.
        self.start_pos=pos
        self.start_rot=rot
        obs = self._env.sim.get_observations_at(pos, rot)
        obs = self._env.step(action={"action": 2})
        obs = self._env.step(action={"action": 3})
        # get current time for calcuting time took to perform one episode
        
        return obs

    def sim_map_to_sim_continuous(self, coords):
        """Converts ground-truth 2D Map coordinates to absolute Habitat
        simulator position and rotation.
        """
        agent_state = self._env.sim.get_agent_state(0)
        y, x = coords
        min_x, min_y = self.map_obj_origin / 100.0

        cont_x = x / 20. + min_x
        cont_y = y / 20. + min_y
        agent_state.position[0] = cont_y
        agent_state.position[2] = cont_x

        rotation = agent_state.rotation
        rvec = quaternion.as_rotation_vector(rotation)

        if self.args.train_single_eps:
            rvec[1] = 0.0
        else:
            rvec[1] = np.random.rand() * 2 * np.pi
        rot = quaternion.from_rotation_vector(rvec)

        return agent_state.position, rot

    def sim_continuous_to_sim_map(self, sim_loc):
        """Converts absolute Habitat simulator pose to ground-truth 2D Map
        coordinates.
        """
        x, y, o = sim_loc
        o = np.rad2deg(o) + 180.0
        return y, x, o
    
    def check_next_data_if_end(self, episode_id):
        # 次のepisodeがepisodeファイルに含まれるならFalse, 
        # 含まれない、つまり全てのepisodeを読み込んでいて、もう次のepisodeがなかったらTrue
        current_scene_len = len(self.eps_data)
        _episode_id = episode_id
        _episode_id += 1
        if _episode_id >= current_scene_len:
            change_scene=True
        else:
            change_scene=False
        return change_scene
    
    def reset_init(self):
        self.eps_data_idx = -1

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        args = self.args
        self.counter += 1
        if (self.counter % args.num_train_episodes == 0) or \
            (self.eps_data_idx >= self.max_eps_no_at_the_scene) or \
            (self.scene_name in ["D7G3Y4RVNrH"]) or \
            (self.counter == 0):
            change_scene = True
        else:
            change_scene = False
        
        if self.args.eval==1:
            _change_scene = self.check_next_data_if_end(self.eps_data_idx)
        change_scene = (change_scene or _change_scene)
        if self.finished:
            self.info["thread_finished"]=True

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []
        self.dists_to_target = []
        if self.info["thread_finished"]==False:
            if change_scene:
                obs = self.switch_scene()
                self.reset_init()
            self.scene_path = self.scene_name
            obs = self.load_new_episode_wo_semmap()
        else:
            rgb = np.zeros(self.rgb_shape).astype(np.uint8)
            depth = np.zeros(self.depth_shape) + 0.1
            state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
            return state, self.info
        rgb = obs[RGB_KEY].astype(np.uint8)
        depth = obs[DEPTH_KEY]
        self.rgb_shape = rgb.shape
        self.depth_shape = depth.shape
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.last_sim_location = self.get_sim_location()
        self.set_metrics_geodesic_distance()
        self.info["dists_to_target_pacman"] = self.dists_to_target
        self.__state_shape__ = state.shape
        self._obs = obs

        # Set info
        self.info["episode_id"]=self.episode["episode_id"]
        self.info["answer"] = self.episode["question"]["answer_text"]
        self.info['question_type'] = self.episode["question"]['question_type']
        self.info["start_position"]=self.episode["start_position"]
        self.info["scene_name"] = self.scene_name
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['goal_cat_id'] = self.goal_idx
        self.info['goal_name'] = self.goal_name
        self.info["question_token"] = self.question_token
        self.info["question_text"] = self.question_text
        self.info["llm_object"] = self.llm_object
        self.info["llm_object_id"] = self.llm_object_id
        self.info["llm_declarative_text"] = self.declarative_text
        self.info["goal_pos"] = self.goal_pos
        self.info["pacman"] = {}
        self.info["invalid"] = False
        self.path_len = 0
        self.info["distance_to_goal"] = self.starting_distance - self.object_boundary
        # save the image on teh start position
        self._save_start_images()
        if (self.dists_to_target[0] < 0 or self.dists_to_target[0] == float("inf")):  # unreachable
            self.info["invalid"]=True
        self.info["observe_action"] = 0

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
        agent_state_before = super().habitat_env.sim.get_agent_state(0) # ADDED
        if action["action"]==0:
            agent_state = super().habitat_env.sim.get_agent_state(0)
            pos = agent_state.position
            rot = agent_state.rotation
            obs = self._env.sim.get_observations_at(pos, rot)
        else:
            obs = self._env.step(action)
        agent_state_after = super().habitat_env.sim.get_agent_state(0) # ADDED
        done = self.get_done(obs)
        rew = np.array(0.)
        rew = rew.astype(np.float64)
        self._obs = obs

        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.set_path_length((dx**2 + dy**2)**0.5)
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        spl, success, dist = 0., 0., 0.
        # if self.NOT_FOUND==False:
        #     _, _, dist = self.get_metrics()
        _, _, dist = self.get_metrics()
        self.info['distance_to_goal'] = dist
        done=False

        rgb = obs[RGB_KEY].astype(np.uint8)
        depth = obs[DEPTH_KEY]
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)

        self.timestep += 1
        self.info['time'] = self.timestep
        self.set_metrics_geodesic_distance()
        self.info["dists_to_target_pacman"] = self.dists_to_target
        if self.args.use_pano_for_itm:
            self.info["sem_contain_goal"] = self.if_sem_contain_goal(obs["semantic_equirectangular"], self.answer_id)
        else:
            self.info["sem_contain_goal"] = self.if_sem_contain_goal(obs[SEMANTIC_KEY], self.answer_id)
        return state, rew, done, self.info
    
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

    def set_metrics_geodesic_distance(self): # goal positionまでの距離を算出して保存
        agent_state = self._env.sim.get_agent_state(0)
        dist_to_target = super().habitat_env.sim.geodesic_distance(
            agent_state.position, self.goal_pos
        )
        self.dists_to_target.append(dist_to_target)

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)

    def get_reward(self, observations):
        curr_loc = self.get_grid_location_mp3d()
        # print(curr_loc,self.gt_planner.fmm_dist.shape)
        self.curr_distance = self.gt_planner.fmm_dist[curr_loc[0],
                                                    curr_loc[1]] / 20.0

        reward = (self.prev_distance - self.curr_distance) * \
            self.args.reward_coeff

        self.prev_distance = self.curr_distance
        return reward
    
    def get_metrics_dtt(self):
        d_T = self.dists_to_target[-1] # agentの最後の位置からgoal pointまでの距離
        d_D = self.dists_to_target[0] - self.dists_to_target[-1]# agentの初期地点からどれだけgoalに近づけたか(値が小さいほどよい)
        d_min = np.array(self.dists_to_target).min()
        d_0 = self.dists_to_target[0]
        return d_T, d_D, d_min, d_0

    def get_metrics(self):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        curr_loc = self.get_grid_location_mp3d()
        dist = self.gt_planner.fmm_dist[curr_loc[0], curr_loc[1]] / 20.0
        if dist == 0.0:
            success = 1
        else:
            success = 0
        spl = min(success * self.starting_distance / self.path_length, 1)
        return spl, success, dist

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

    def get_grid_location_mp3d(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        map_loc = maps.to_grid(
            agent_state.position[2], agent_state.position[0], self.grid_dimensions, pathfinder=self._env.sim.pathfinder
        )
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return map_loc[0], map_loc[1], o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location() # get current location in meter
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do
    
    def set_path_length(self, length):
        self.path_len += length

