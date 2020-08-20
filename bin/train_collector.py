import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from baselines.bench.monitor import load_results
from multiagent.matrpo import MATRPO
from multiagent.plot import plot

env_id = 'collector'
model = 'matrpo'
seed = 1
reward_path = '/home/lihepeng/Documents/Github/results/training/{}/{}/s{}'.format(env_id, model, seed)
load_path = '/home/lihepeng/Documents/Github/results/graphs/{}/{}/s{}'.format(env_id, model, seed)
network_kwargs = {'num_layers': 2, 'num_hidden': 128, 'activation': 'selu'}
agents = MATRPO(
    env_id=env_id,
    nsteps=1000,
    network='mlp',
    num_env=10,
    admm_iter=200,
    ent_coef=0.1,
    load_path=load_path,
    logger_dir=reward_path,
    seed=seed,
    info_keywords=tuple('r{}'.format(i) for i in range(8)),
    mode='matrpo',
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



