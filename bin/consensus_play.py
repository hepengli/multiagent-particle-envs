import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from multiagent.matrpo import MATRPO

seed = 1
mode = 'trpo'
env_id = 'consensus'
load_path = '/home/lihepeng/Documents/Github/results/graphs/{}/{}/s{}'.format(env_id, mode, seed)
network_kwargs = {'num_layers': 2, 'num_hidden': 128, 'activation': 'selu'}
agents = MATRPO(
    env_id=env_id,
    nsteps=1000,
    network='mlp',
    num_env=1,
    seed=seed,
    finite=True,
    admm_iter=0,
    load_path=load_path,
    info_keywords=tuple('r{}'.format(i) for i in range(3)),
    mode=mode,
    **network_kwargs)

agents.play()

# dists = np.zeros([1000, 100])
# collisions = np.zeros([1000, 100])
# agents.model.test = True
# for ep in range(1000):
#     obs_n = agents.test_env.reset()
#     i = 0
#     while True:
#         # query for action from each agent's policy
#         act_n, _, _ = agents.model.step(obs_n)
#         # step environment
#         obs_n, reward_n, done_n, info_n = agents.test_env.step(act_n)
#         dists[ep, i] = info_n['n'][0][2]
#         collisions[ep, i] = sum([info[1] for info in info_n['n']])
#         i += 1
#         # break
#         if done_n:
#             print(ep)
#             print('done!')
#             break

# ep_dists = np.mean(dists, axis=1)
# ep_collisions = np.mean(collisions, axis=1)

# print(ep_dists.mean(), ep_dists.std())
# print(ep_collisions.mean(), ep_collisions.std())

