import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from baselines.bench.monitor import load_results
from multiagent.matrpo import MATRPO
from multiagent.plot import plot

def myplot(adv_rewards, good_rewards):
    plt.close()
    fig = plt.figure()
    ax = plt.subplot(211)
    ax.plot(adv_rewards)
    ax.grid(axis='y')
    ax.set_title('Speaker\'s rewards')
    ax1 = plt.subplot(212)
    ax1.plot(good_rewards)
    ax1.grid(axis='y')
    ax1.set_title('Listener\'s rewards')
    plt.tight_layout(rect=(0,0,1,1))
    plt.show(block=False)
    plt.pause(0.5)

def main():
    env_id = 'simple_world_comm'
    model = 'matrpo'
    seed = 2
    network_kwargs = {'num_layers': 2, 'num_hidden': 128, 'activation': 'selu'}
    reward_path = '/home/lihepeng/Documents/Github/multiagent-particle-envs/results/training/{}/{}/s{}'.format(env_id, model, seed)
    load_path = '/home/lihepeng/Documents/Github/multiagent-particle-envs/results/graphs/{}/{}/s{}'.format(env_id, model, seed)
    agents = MATRPO(
        env_id=env_id,
        nsteps=2000,
        network='mlp',
        num_env=10,
        admm_iter=[0,80],
        ob_clip_range=5.0,
        load_path=load_path,
        logger_dir=reward_path,
        seed=seed,
        info_keywords=tuple('r{}'.format(i) for i in range(7)),
        adv='independent',
        agt='cooperative',
        **network_kwargs)

    # training
    adv_rewards, good_rewards = np.array([]), np.array([])
    total_timesteps = 500
    for step in range(1, total_timesteps+1):
        actions, obs, returns, dones, values, advs, neglogpacs, monitor_rewards = agents.runner.run()
        agents.model.train(actions, obs, returns, dones, values, advs, neglogpacs)
        adv_rewards = np.hstack([adv_rewards, np.mean(np.sum(monitor_rewards[:agents.n_adv], axis=0))])
        good_rewards = np.hstack([good_rewards, np.mean(np.sum(monitor_rewards[agents.n_adv:], axis=0))])
        myplot(adv_rewards, good_rewards)

        # df_train = load_results(reward_path)
        # plot(df_train, agents)
        if step % 10 == 0:
            agents.model.save()

            # # play
            # agents.play()
            # # pass

if __name__ == "__main__":
    main()


