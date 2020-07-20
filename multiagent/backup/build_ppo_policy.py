import tensorflow as tf
from gym import spaces
from multiagent.agent_ppo_model import AgentModel
from baselines.common.models import get_network_builder
from baselines.common.mpi_adam import MpiAdam
from baselines.common.policies import PolicyWithValue

def Policy(env, world, network, nsteps, rho=1.0, ent_coef=0.0, vf_coef=.5,
           max_grad_norm=.5, seed=None, load_path=None, **network_kwargs):
    policies = []
    for index, agent in enumerate(world.agents):
        agent = world.agents[index]
        agent.id = index
        agent.comm_matrix = world.comm_matrix[world.comm_matrix[:,index]!=0]
        agent.observation_space = env.observation_space[index]
        agent = create_action_space(agent, env, world)
        with tf.name_scope(agent.name):
            model = AgentModel(agent, network, nsteps, rho, ent_coef, vf_coef, max_grad_norm, 
                               seed, load_path, **network_kwargs)
            policies.append(model)

    return policies

def create_action_space(agent, env, world):
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
            agent.nmates = agent.action_space.nvec.size
            agent.nlogits = agent.action_space.nvec.sum()
        else:
            agent.action_space = spaces.Tuple(all_action_spaces)
            agent.nmates = agent.action_space.__len__()
            agent.nnlogits = None
    else:
        agent.action_space = all_action_spaces[0]
        agent.nmates = 1
        agent.nnlogits = agent.action_space.n
    
    return agent