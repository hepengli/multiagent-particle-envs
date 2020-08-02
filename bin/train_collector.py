import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from baselines.bench.monitor import load_results
from multiagent.matrpo import MATRPO
from multiagent.plot import plot

def main():
    env_id = 'collector'
    model = 'matrpo'
    seed = 2
    reward_path = '/home/lihepeng/Documents/Github/results/training/{}/{}/s{}'.format(env_id, model, seed)
    load_path = '/home/lihepeng/Documents/Github/results/graphs/{}/{}/s{}'.format(env_id, model, seed)
    network_kwargs = {'num_layers': 3, 'num_hidden': 128}
    agents = MATRPO(
        env_id=env_id,
        nsteps=1000,
        network='mlp',
        num_env=15,
        admm_iter=200,
        # ob_clip_range=5.0,
        max_kl=0.01,
        load_path=load_path,
        logger_dir=reward_path,
        seed=seed,
        finite=True,
        gamma=0.95,
        info_keywords=tuple('r{}'.format(i) for i in range(8)),
        adv='cooperative',
        **network_kwargs)

    # training
    total_timesteps = 1000
    for step in range(1, total_timesteps+1):
        actions, obs, returns, dones, values, advs, neglogpacs = agents.runner.run()
        agents.model.train(actions, obs, returns, dones, values, advs, neglogpacs)

        df_train = load_results(reward_path)
        plot(df_train, agents, 150)
        if step % 10 == 0:
            agents.model.save()

if __name__ == "__main__":
    main()


