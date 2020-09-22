import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from multiagent.matrpo import MATRPO

seed = 1
mode = 'central'
env_id = 'collector'
load_path = '/home/lihepeng/Documents/Github/results/graphs/{}/{}/s{}'.format(env_id, mode, seed)
network_kwargs = {'num_layers': 2, 'num_hidden': 128, 'activation': 'selu'}
agents = MATRPO(
    env_id=env_id,
    nsteps=1000,
    network='mlp',
    num_env=1,
    seed=seed,
    finite=False,
    admm_iter=0,
    load_path=load_path,
    info_keywords=tuple('r{}'.format(i) for i in range(8)),
    mode=mode,
    **network_kwargs)

num_episodes = 1000
collected_treasures = np.zeros([num_episodes, 100])
deposited_treasures = np.zeros([num_episodes, 100])
collisions = np.zeros([num_episodes, 100])
agents.model.test = True
for ep in range(num_episodes):
    obs_n = agents.test_env.reset()
    i = 0
    while True:
        # query for action from each agent's policy
        act_n, _, _ = agents.model.step(obs_n)
        # step environment
        obs_n, reward_n, done_n, info_n = agents.test_env.step(act_n)
        collected_treasures[ep, i] = sum([info[0] for info in info_n['n']])
        deposited_treasures[ep, i] = sum([info[1] for info in info_n['n']])
        collisions[ep, i] = sum([info[2] for info in info_n['n'][:10]])/2
        i += 1
        # break
        if done_n:
            print(ep)
            print('done!')
            break

ep_deposited_treasures = np.sum(deposited_treasures, axis=1)
ep_collected_treasures = np.sum(collected_treasures, axis=1)
ep_collisions = np.sum(collisions, axis=1)

print(ep_deposited_treasures.mean(), ep_deposited_treasures.std())
print(ep_collected_treasures.mean(), ep_collected_treasures.std())
print(ep_collisions.mean(), ep_collisions.std())

