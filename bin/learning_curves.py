import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from baselines.bench.monitor import load_results

def get_training_curve_1(model):
    env_id = 'simple_catchup'
    seeds = [1,2,3,4,5]
    pred_results = []
    prey_results = []
    for seed in seeds:
        reward_path = '/home/lihepeng/Documents/Github/multiagent-particle-envs/results/training/{}/{}/s{}'.format(env_id, model, seed)
        df_train = load_results(reward_path)
        pred_data = df_train[['r0','r1','r2']].sum(axis=1).values
        pred_data = np.mean(pred_data.reshape(-1, 100), axis=1)
        pred_results.append(pred_data)

        prey_data = df_train[['r3']].sum(axis=1).values
        prey_data = np.mean(prey_data.reshape(-1, 100), axis=1)
        prey_results.append(prey_data)

    return np.array(pred_results), np.array(prey_results)


def get_training_curve(model):
    env_id = 'simple_spread_6'
    seeds = [1,2,3,4,5]
    results = []
    for seed in seeds:
        reward_path = '/home/lihepeng/Documents/Github/multiagent-particle-envs/results/training/{}/{}/s{}'.format(env_id, model, seed)
        df_train = load_results(reward_path)
        data = df_train[['r0','r1','r2','r3','r4','r5']].mean(axis=1).values
        data = np.mean(data.reshape(-1, 100), axis=1)
        results.append(data)

    return np.array(results)

xdata = np.arange(200)
models = ['ctrpo','matrpo','trpo']
results = []
for model in models:
    results.append(get_training_curve(model))

fig = plt.figure()
sns.tsplot(time=xdata, data=results[0], color='r', linestyle='-')
sns.tsplot(time=xdata, data=results[1], color='g', linestyle='-')
sns.tsplot(time=xdata, data=results[2], color='b', linestyle='-')
plt.show()
