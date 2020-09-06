import numpy as np
import tensorflow as tf
from multiagent.agent_model import AgentModel
from gym import spaces

def Policy(env, world, network, nbatch, mode, rho, max_kl, ent_coef, vf_stepsize, vf_iters,
           cg_damping, cg_iters, lbfgs_iters, load_path, **network_kwargs):
    policies = []
    for index, agent in enumerate(world.agents):
        agent = world.agents[index]
        agent.id = index
        agent.comm_matrix = world.comm_matrix[world.comm_matrix[:,index]!=0]
        agent.comms = agent.comm_matrix[:,index][:,None]
        agent.neighbors = [id for id in np.where(agent.comm_matrix!=0)[1] if id != index]
        agent.observation_space = env.observation_space[index]
        if mode == 'matrpo':
            agent = cooperative_action_space(agent, env, world)
        elif mode == 'trpo':
            agent = independent_action_space(agent, env, world)
        else:
            agent = cooperative_action_space(agent, env, world)
        model = AgentModel(agent, network, nbatch, rho, max_kl, ent_coef, vf_stepsize, vf_iters,
                           cg_damping, cg_iters, lbfgs_iters, load_path, **network_kwargs)
        policies.append(model)
    
    return policies

def cooperative_action_space(agent, env, world):
    env.discrete_action_input = True
    all_action_spaces = []
    for i, other in enumerate(world.agents):
        # store agent's action index
        if other is agent:
            agent.action_index = [len(all_action_spaces)]
        # Get parterners' action spaces
        if (hasattr(other, 'adversary') and (other.adversary==agent.adversary)) or \
        (not hasattr(other, 'adversary')):
            if agent.silent or isinstance(env.action_space[i], spaces.Discrete): # no communication action
                all_action_spaces.append(env.action_space[i])
            else:
                if isinstance(env.action_space[i], spaces.Tuple):
                    all_action_spaces.extend(env.action_space[i].spaces)
                else: # must be MultiDicrete space
                    h0, h1 = env.action_space[i].high
                    all_action_spaces.extend([spaces.Discrete(h0), spaces.Discrete(h1)])
                    if other is agent:
                        agent.action_index = agent.action_index + [agent.action_index[0]+1]

    # Augment agent's action space to include parterners' action spaces
    if len(all_action_spaces) > 1:
        # If all action spaces are discrete, simplify to MultiDiscrete action space
        if all([isinstance(ac_space, spaces.Discrete) for ac_space in all_action_spaces]):
            agent.action_space = spaces.MultiDiscrete([ac_space.n for ac_space in all_action_spaces])
            agent.nmates = agent.action_size = agent.action_space.nvec.size
        else:
            agent.action_space = spaces.Tuple(all_action_spaces)
            agent.nmates = agent.action_size = agent.action_space.__len__()
    else:
        agent.action_space = all_action_spaces[0]
        agent.nmates = agent.action_size = 1
    
    return agent


def independent_action_space(agent, env, world):
    env.discrete_action_input = True
    for i, other in enumerate(world.agents):
        # store agent's action index
        if other is agent:
            agent.action_index = i
            agent.action_space = env.action_space[i]
            agent.nmates = 1

    return agent