import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from baselines.bench.monitor import load_results
from multiagent.matrpo import MATRPO
from multiagent.plot import plot

env_id = 'collector'
mode = 'central'
seed = 3
reward_path = '/home/lihepeng/Documents/Github/results/training/{}/{}/s{}'.format(env_id, mode, seed)
load_path = '/home/lihepeng/Documents/Github/results/graphs/{}/{}/s{}'.format(env_id, mode, seed)
network_kwargs = {'num_layers': 3, 'num_hidden': 128, 'activation': 'selu'}
agents = MATRPO(
    env_id=env_id,
    seed=seed,
    num_env=10,
    nsteps=1000,
    admm_iter=200,
    network='mlp',
    load_path=load_path,
    logger_dir=reward_path,
    info_keywords=tuple('r{}'.format(i) for i in range(8)),
    mode=mode,
    **network_kwargs)

# training
total_timesteps = 1000
for step in range(1, total_timesteps+1):
    actions, obs, returns, dones, values, advs, neglogpacs = agents.runner.run()
    agents.model.train(actions, obs, returns, dones, values, advs, neglogpacs)

    df_train = load_results(reward_path)
    plot(df_train, agents, 100)
    if step % 10 == 0:
        agents.model.save()



