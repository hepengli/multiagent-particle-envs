import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from multiagent.matrpo import MATRPO

def main():
    env_id = 'simple_predator_prey'
    adv = 'independent'
    agt = 'independent'
    model = adv+'_vs_'+agt
    seed = 5
    network_kwargs = {'num_layers': 2, 'num_hidden': 128, 'activation': 'selu'}
    load_path = '/home/lihepeng/Documents/Github/results/graphs/{}/{}/s{}'.format(env_id, model, seed)
    agents = MATRPO(
        env_id=env_id,
        nsteps=2000,
        network='mlp',
        num_env=1,
        admm_iter=[0,0],
        ob_clip_range=5.0,
        load_path=load_path,
        seed=seed,
        info_keywords=tuple('r{}'.format(i) for i in range(7)),
        adv=adv,
        agt=agt,
        **network_kwargs)

    agents.play()

if __name__ == "__main__":
    main()

