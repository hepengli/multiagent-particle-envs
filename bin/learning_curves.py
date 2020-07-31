import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from baselines.bench.monitor import load_results

def get_training_curve(model, num_agents=3):
    env_id = 'simple_spread_{}'.format(num_agents)
    agents = ['r{}'.format(i) for i in range(num_agents)]
    seeds = [1,2,3,4,5]
    results = []
    for seed in seeds:
        reward_path = '/home/lihepeng/Documents/Github/results/training/{}/{}/s{}'.format(env_id, model, seed)
        df_train = load_results(reward_path)
        data = df_train[agents].mean(axis=1).values
        data = np.mean(data.reshape(-1, 100), axis=1)
        results.append(data)

    return np.array(results)

xdata = np.arange(200)
models = ['ctrpo','matrpo','trpo']
results = []
for model in models:
    results.append(get_training_curve(model, num_agents=6))

fig = plt.figure()
sns.tsplot(time=xdata, data=results[0], color='r', linestyle='-', legend=True)
sns.tsplot(time=xdata, data=results[1], color='g', linestyle='-', legend=True)
sns.tsplot(time=xdata, data=results[2], color='b', linestyle='-', legend=True)
plt.legend(labels=['Central TRPO', 'MATRPO', 'Independent TRPO'])
plt.show()


# Predator prey
def get_training_curve_1(model):
    env_id = 'simple_predator_prey'
    seeds = [1,2,3]
    pred_results = []
    prey_results = []
    for seed in seeds:
        reward_path = '/home/lihepeng/Documents/Github/results/training/{}/{}/s{}'.format(env_id, model, seed)
        df_train = load_results(reward_path)
        pred_data = df_train[['r0','r1','r2','r3']].sum(axis=1).values[:500*150]
        pred_data = np.mean(pred_data.reshape(-1, 150), axis=1)
        pred_results.append(pred_data)

        prey_data = df_train[['r4','r5','r6']].sum(axis=1).values[:500*150]
        prey_data = np.mean(prey_data.reshape(-1, 150), axis=1)
        prey_results.append(prey_data)

    return np.array(pred_results), np.array(prey_results)


xdata = np.arange(500)
models = ['independent_vs_independent']
pred_results, prey_results = [], []
for model in models:
    pred, prey = get_training_curve_1(model)
    pred_results.append(pred)
    prey_results.append(prey)

fig = plt.figure()
sns.tsplot(time=xdata, data=pred_results[0], color='r', linestyle='-', legend=True)
# sns.tsplot(time=xdata, data=results[1], color='g', linestyle='-', legend=True)
# sns.tsplot(time=xdata, data=results[2], color='b', linestyle='-', legend=True)
# plt.legend(labels=['Central TRPO', 'MATRPO', 'Independent TRPO'])
plt.show()

