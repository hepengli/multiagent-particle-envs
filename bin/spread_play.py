import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from multiagent.matrpo import MATRPO

def main():
    seed = 1
    env_id = 'simple_spread'
    model = 'trpo'
    network_kwargs = {'num_layers': 2, 'num_hidden': 128, 'activation': 'selu'}
    load_path = '/home/lihepeng/Documents/Github/results/graphs/{}/{}/s{}'.format(env_id+'_3', model, seed)
    agents = MATRPO(
        env_id=env_id,
        nsteps=1000,
        network='mlp',
        num_env=1,
        finite=False,
        gamma=0.95,
        ob_clip_range=5.0,
        ob_normalization=True,
        admm_iter=[0,0],
        load_path=load_path,
        adv='independent',
        info_keywords=tuple('r{}'.format(i) for i in range(3)),
        **network_kwargs)

    agents.play()

if __name__ == "__main__":
    main()
