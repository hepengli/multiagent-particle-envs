import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from baselines.bench.monitor import load_results
from multiagent.matrpo import MATRPO
from multiagent.plot import plot

seed = 5
mode = 'matrpo'
env_id = 'simple_spread'
reward_path = '/home/lihepeng/Documents/Github/results/training/{}/{}/s{}'.format(env_id, mode, seed)
load_path = '/home/lihepeng/Documents/Github/results/graphs/{}/{}/s{}'.format(env_id, mode, seed)
network_kwargs = {'num_layers': 2, 'num_hidden': 128, 'activation': 'selu'}
agents = MATRPO(
    env_id=env_id,
    seed=seed,
    num_env=10,
    nsteps=1000,
    max_kl=0.003,
    network='mlp',
    finite=False,
    admm_iter=100,
    load_path=load_path,
    logger_dir=reward_path,
    info_keywords=tuple('r{}'.format(i) for i in range(3)),
    mode=mode,
    **network_kwargs)

# training
total_timesteps = 500
for step in range(1, total_timesteps+1):
    actions, obs, returns, dones, values, advs, neglogpacs = agents.runner.run()
    agents.model.train(actions, obs, returns, dones, values, advs, neglogpacs)

    df_train = load_results(reward_path)
    plot(df_train, agents, 100)
    if step % 10 == 0:
        agents.model.save()

