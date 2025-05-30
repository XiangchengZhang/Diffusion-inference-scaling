'''
Evaluation script for original QGPO time dependent guidance
'''
import os
import gym
import d4rl
import scipy
import tqdm
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusion_SDE.loss import loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet
from utils import get_args, pallaral_eval_policy
from dataset.dataset import D4RL_dataset


LOAD_FAKE=False

args = get_args()
args.actor_load_path = './models_rl/' + args.env + '0large_actor/behavior_ckpt600.pth'
args.q_load_path = './models_rl/' + args.env + '0large_actor/critic_ckpt100.pth'

for dir in ["./models_rl", "./logs"]:
    if not os.path.exists(dir):
        os.makedirs(dir)
if not os.path.exists(os.path.join("./models_rl", str(args.expid))):
    os.makedirs(os.path.join("./models_rl", str(args.expid)))
# writer = SummaryWriter("./logs/" + str(args.expid))

env = gym.make(args.env)
# env.seed(args.seed)
# env.action_space.seed(args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# args.eval_func = functools.partial(pallaral_eval_policy, env_name=args.env, seed=args.seed, eval_episodes=args.seed_per_evaluation, diffusion_steps=args.diffusion_steps)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
# args.writer = writer
args.s = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)
args.marginal_prob_std_fn = marginal_prob_std_fn
score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
score_model.q[0].to(args.device)
actor_ckpt = torch.load(args.actor_load_path, map_location=args.device)
score_model.load_state_dict(actor_ckpt)
q_ckpt = torch.load(args.q_load_path, map_location=args.device)
score_model.q[0].load_state_dict(q_ckpt, strict=False)
epoch = 0


for guidance_scale in args.s:
    for seed in range(5):
        args.seed = seed
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        score_model.q[0].guidance_scale = guidance_scale
        args.eval_func = functools.partial(pallaral_eval_policy, env_name=args.env, seed=args.seed, eval_episodes=args.seed_per_evaluation, diffusion_steps=args.diffusion_steps)
        envs = args.eval_func(score_model.select_actions)
        mean = np.mean([envs[i].buffer_return for i in range(args.seed_per_evaluation)])
        std = np.std([envs[i].buffer_return for i in range(args.seed_per_evaluation)])
        print('seed: {}, mean: {}, std: {}'.format(seed, mean, std))