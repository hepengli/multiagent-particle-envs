from gym.envs.registration import register
from multiagent.environment import MultiAgentEnv

# Multiagent envs
# ----------------------------------------
register(
   	id='MultiAgent-v0',
   	entry_point='multiagent:MultiAgentEnv',
    max_episode_steps=100,
)
