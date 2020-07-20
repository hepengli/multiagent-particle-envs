import pandas as pd
import matplotlib.pyplot as plt

def plot(df_train, agents):
    rolling_window = 50
    adversaries = ['r{}'.format(i) for i in range(agents.n_adv)]
    adversaries_rewards = pd.Series(df_train[adversaries].sum(axis=1)).rolling(rolling_window)
    adversaries_rewards = adversaries_rewards.mean().values
    if agents.n_agt == 0:
        plt.close()
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(adversaries_rewards)
        ax.set_title('Rewards')
        plt.tight_layout(rect=(0,0,1,1))
        plt.show(block=False)
        plt.pause(0.5)
    else:
        good_agents = ['r{}'.format(i) for i in range(agents.n_adv, agents.n)]
        good_agents_rewards = pd.Series(df_train[good_agents].sum(axis=1)).rolling(rolling_window)
        good_agents_rewards = good_agents_rewards.mean()
        # plot
        plt.close()
        fig = plt.figure()
        ax = plt.subplot(211)
        ax.plot(adversaries_rewards)
        ax.set_title('Adversaries\' rewards')
        ax1 = plt.subplot(212)
        ax1.plot(good_agents_rewards)
        ax1.set_title('Good agent\'s rewards')
        plt.tight_layout(rect=(0,0,1,1))
        plt.show(block=False)
        plt.pause(0.5)
