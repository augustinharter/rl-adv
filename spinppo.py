#%%
from spinup import ppo_pytorch
import gym
import mujoco_py as mjp
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym.spaces import Box, Discrete
from spinup.utils.test_policy import load_policy_and_env, run_policy

class CategoricalPi(nn.Module):
    def __init__(self, obs_dim, act_dim, activation):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dist = T.distributions.Categorical
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, act_dim),
            nn.Softmax())

    def get_log_p(self, pi, actions):
        return pi.log_prob(actions)

    def forward(self, obs, actions):
        pi = self.dist(self.model(obs))
        logp_a = None
        if actions is not None:
            logp_a = self.get_log_p(pi, actions)
        return pi, logp_a

class GaussianPi(nn.Module):
    def __init__(self, obs_dim, act_dim, activation):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dist = T.distributions.Normal
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = T.nn.Parameter(T.as_tensor(log_std))
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            activation(), 
            nn.Linear(64, 64),
            activation(), 
            nn.Linear(64, act_dim))

    def get_log_p(self, pi, actions):
        return pi.log_prob(actions).sum(axis=-1)

    def forward(self, obs, actions):
        mu = self.model(obs)
        std = T.exp(self.log_std)
        pi = self.dist(mu, std)
        logp_a = None
        if actions is not None:
            logp_a = self.get_log_p(pi, actions)
        return pi, logp_a

class V(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.model = nn.Sequential(
            nn.Linear(obs_dim,64),
            nn.LeakyReLU(0.1),
            nn.Linear(64,32),
            nn.LeakyReLU(0.1),
            nn.Linear(32,1))
    def forward(self, obs):
        return self.model(obs).flatten()

class CustomActorCritic(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            self.pi = GaussianPi(obs_dim, act_space.shape[0], nn.Tanh)
        elif isinstance(act_space, Discrete):
            self.pi = CategoricalPi(obs_dim, act_space.n, nn.Tanh)

        self.v = V(obs_dim)
    
    def step(self, obs):
        with T.no_grad():
            v = self.v(obs)
            pi, _ = self.pi(obs, None)
            a = pi.sample()
            logp_a = self.pi.get_log_p(pi, a)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.pi(obs, None)[0].sample()

def train(xprmt_path):
    env_fn = lambda: gym.make("Swimmer-v2")
    #env_fn = lambda: gym.make("Ant-v2")
    ppo_pytorch(env_fn, actor_critic=CustomActorCritic, logger_kwargs={"output_dir":xprmt_path}, epochs=1000, steps_per_epoch=4000,
        pi_lr=3e-3, gamma=0.99)
    #ppo_pytorch(env_fn, logger_kwargs={"output_dir":xprmt_path})

#%%
if __name__ == "__main__":
    xprmt_path = "./experiments/swimmer02"
    #train(xprmt_path)
    env, get_action = load_policy_and_env(xprmt_path)
    run_policy(env, get_action)

# %%
