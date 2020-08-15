import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from multiagent.matrpo import MATRPO

def main():
    seed = 3
    env_id = 'simple_predator_prey'
    model = 'independent_vs_independent'
    network_kwargs = {'num_layers': 2, 'num_hidden': 128, 'activation': 'tanh'}
    load_path = '/home/lihepeng/Documents/Github/results/graphs/{}/{}/s{}'.format(env_id, model, seed)
    agents = MATRPO(
        env_id=env_id,
        nsteps=1000,
        num_env=1,
        admm_iter=[0,0],
        load_path=load_path,
        adv='independent',
        agt='independent',
        network='mlp',
        **network_kwargs)

    agents.play()

if __name__ == "__main__":
    main()

