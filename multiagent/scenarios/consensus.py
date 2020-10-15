import numpy as np
from multiagent.new_core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario
from scipy.linalg import toeplitz


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 5
        num_walls = 4
        # add comm network
        world.comm_matrix = toeplitz(
            [1]+[0]*(num_agents-2), 
            [1,-1]+[0]*(num_agents-2)
        ).astype(np.float32)
        world.comm_matrix = np.vstack([
            world.comm_matrix,
            np.array([[-1]+[0]*(num_agents-2)+[1]]),
        ]).astype(np.float32)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.ghost = True
            agent.size = 0.05
        # add walls
        world.walls = [Wall() for i in range(num_walls)]
        for i, landmark in enumerate(world.walls):
            landmark.name = 'wall %d' % i
            landmark.orient = 'H' if i % 2 == 0 else 'V'
            landmark.axis_pos = - 1.2 if i < 2 else 1.2
            landmark.width = 0.4
            landmark.endpoints = (-1.2, 1.2)
        # make initial conditions
        self.reset_world(world, np.random)
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_his_pos = np.tile(agent.state.p_pos, [10,1])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_his_vel = np.tile(agent.state.p_vel, [10,1])
            agent.state.c = np.zeros(world.dim_c)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if a is not agent and self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def neighbors(self, agent, world):
        edges = np.where(world.comm_matrix[:, agent.id] != 0)[0]
        nbs = np.unique(np.hstack([np.where(world.comm_matrix[e]!=0)[0] for e in edges]))

        return [world.agents[nb] for nb in nbs if nb != agent.id]

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        for agent in world.agents:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in self.neighbors(agent, world)]
            rew -= sum(dists) * 0.1

            def bound(x):
                if x < 0.9:
                    return 0
                if x < 1.0:
                    return (x - 0.9) * 10
                return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)

            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                rew -= 2 * bound(x)

        return rew

    # def observation(self, agent, world):
    #     # communication of all other agents
    #     comm = []
    #     other_pos = []
    #     other_vel = []
    #     other_his_pos = []
    #     other_his_vel = []
    #     # for other in world.agents:
    #     for other in self.neighbors(agent, world):
    #         if other is agent: continue
    #         comm.append(other.state.c)
    #         other_pos.append(other.state.p_pos - agent.state.p_pos)
    #         other_vel.append(other.state.p_vel - agent.state.p_vel)
    #         other_his_pos.append(other.state.p_his_pos - agent.state.p_his_pos)
    #         other_his_vel.append(other.state.p_his_vel - agent.state.p_his_vel)
    #     # ob = np.concatenate(other_vel + other_pos)
    #     ob = np.concatenate(other_his_vel + other_his_pos).ravel()

    #     return ob.astype(np.float32)

    def observation(self, agent, world):
        # communication of all other agents
        comm = []
        his_len = 10
        other_pos_diff = []
        other_vel_diff = []
        other_pos_diff_int = [] # integral
        other_vel_diff_int = []
        # for other in world.agents:
        for other in self.neighbors(agent, world):
            if other is agent: continue
            comm.append(other.state.c)
            other_pos_diff.append(other.state.p_his_pos[-his_len:] - agent.state.p_his_pos[-his_len:])
            other_vel_diff.append(other.state.p_his_vel[-his_len:] - agent.state.p_his_vel[-his_len:])
            other_pos_diff_int.append(np.sum(other.state.p_his_pos - agent.state.p_his_pos, axis=0))
            other_vel_diff_int.append(np.sum(other.state.p_his_vel - agent.state.p_his_vel, axis=0))
        ob = np.concatenate(other_pos_diff + other_vel_diff).ravel()
        # ob_2 = np.concatenate(other_pos_diff_int + other_vel_diff_int)
        # ob = np.concatenate([ob_1, ob_2]).ravel()

        return ob.astype(np.float32)
