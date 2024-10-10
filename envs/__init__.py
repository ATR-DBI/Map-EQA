import torch
import numpy as np

from .habitat import construct_eqa_envs, make_env_eqa_fn, make_env_eqa_vqa_fn


def make_vec_eqa_envs(args, make_env_fn=make_env_eqa_fn):
    envs = construct_eqa_envs(args, make_env_fn)
    envs = VecPyTorch(envs, args.device)
    return envs

def make_vec_eqa_vqa_envs(args, make_env_fn=make_env_eqa_vqa_fn):
    envs = construct_eqa_envs(args, make_env_fn)
    envs = VecPyTorch(envs, args.device)
    return envs


# Adapted from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device, multi=True):
        self.venv = venv
        if multi:
            self.num_envs = venv.num_envs
        else:
            self.num_envs=1
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.device = device

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        obs, reward, done, info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def get_obs(self):
        obs = self.venv.get_obs()
        return obs

    def get_frontier_map(self, planner_inputs):
        result = self.venv.get_frontier_map(planner_inputs)
        return result
    
    def get_rewards_one_thread(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(np.array(reward)).float()
        return reward

    def plan_act_and_preprocess(self, inputs):
        obs, reward, done, info = self.venv.plan_act_and_preprocess(inputs)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(np.array(reward)).float()
        return obs, reward, done, info
    
    def plan_act_and_preprocess_one_thread(self, inputs):
        obs, reward, done, info = self.venv.plan_act_and_preprocess(inputs)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(np.array(reward)).float()
        return obs, reward, done, info

    def plan_act_and_preprocess_wo_reward(self, inputs):
        obs, done, info = self.venv.plan_act_and_preprocess_wo_reward(inputs)
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, done, info

    def close(self):
        return self.venv.close()
