import numpy as np
from multiagent.new_core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.cache_dists = True
        world.dim_c = 3
        #world.damping = 1
        num_good_agents = 4
        num_adversaries = 3
        num_agents = num_adversaries + num_good_agents
        num_barrier = 1
        num_food = 3
        num_forests = 2
        num_walls = 4

        # add comm network
        world.comm_matrix = np.array([
            [1.,-1., 0., 0., 0., 0., 0.],
            [0., 1.,-1., 0., 0., 0., 0.],
            [0., 0., 0., 1.,-1., 0., 0.],
            [0., 0., 0., 0., 1.,-1., 0.],
            [0., 0., 0., 0., 0., 1.,-1.],
        ], dtype=np.float32)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.i = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.leader = True if (i == 0) or (i == num_adversaries) else False
            agent.silent = True
            agent.holding = False
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.065 if agent.adversary else 0.045
            agent.initial_mass = 2.25 if agent.adversary else 1.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.barrier = [Landmark() for i in range(num_barrier)]
        for i, landmark in enumerate(world.barrier):
            landmark.i = i + num_agents
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        world.foods = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.foods):
            landmark.i = i + num_agents + num_barrier
            landmark.name = 'food %d' % i
            landmark.alive = True
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.respawn_prob = 1.0
            landmark.boundary = False
        world.forests = [Landmark() for i in range(num_forests)]
        for i, landmark in enumerate(world.forests):
            landmark.i = i + num_agents + num_barrier + num_food
            landmark.name = 'forest %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.25
            landmark.boundary = False
        world.walls = [Wall() for i in range(num_walls)]
        for i, landmark in enumerate(world.walls):
            landmark.name = 'wall %d' % i
            landmark.orient = 'H' if i % 2 == 0 else 'V'
            landmark.axis_pos = - 1.2 if i < 2 else 1.2
            landmark.width = 0.4
            landmark.endpoints = (-1.2, 1.2)
        world.landmarks += world.barrier
        world.landmarks += world.foods
        world.landmarks += world.forests
        # make initial conditions
        self.reset_world(world, np.random)
        return world

    def post_step(self, world):
        leaders = [ga for ga in self.good_agents(world) if ga.leader]
        members = [ga for ga in self.good_agents(world) if not ga.leader]
        for f in world.foods:
            if f.alive:
                for m in members:
                    if not m.holding and self.is_collision(f, m, world):
                        f.alive = False
                        m.holding = True
                        m.color = np.array([0.55, 0.55, 0.85])
                        f.state.p_pos = np.array([-999., -999.])
                        break
            else:
                if np.random.uniform() <= f.respawn_prob:
                    bound = 0.95
                    f.state.p_pos = np.random.uniform(low=-bound, high=bound,
                                                      size=world.dim_p)
                    f.alive = True
        for m in members:
            if m.holding:
                for l in leaders:
                    if self.is_collision(m, l, world):
                        m.holding = False
                        l.color = m.color - np.array([0.35, 0.35, 0.35])
                        m.color = np.array([0.85, 0.85, 0.85], dtype=np.float32)

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.95, 0.45, 0.45]) if agent.adversary else np.array([0.85, 0.85, 0.85])
            agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            # random properties for landmarks
        for i, landmark in enumerate(world.barrier):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.foods):
            landmark.color = np.array([0.15, 0.15, 0.65])
        for i, landmark in enumerate(world.forests):
            landmark.color = np.array([0.6, 0.9, 0.6])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.holding = False
        for i, landmark in enumerate(world.barrier):
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.forests):
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        def is_collision(agent1, agent2):
            delta_pos = agent1.state.p_pos - agent2.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = agent1.size + agent2.size
            return True if dist < dist_min else False
        for i, landmark in enumerate(world.foods):
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.alive = True
            while any([is_collision(landmark, barrier) for barrier in world.barrier+world.forests]):
                landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)

        world.calculate_distances()

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent, world):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2, world):
        dist = world.cached_dist_mag[agent1.i, agent2.i]
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        # boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False

    def agent_reward(self, agent, world):
        rew = 0
        adversaries = self.adversaries(world)
        leaders = [ga for ga in self.good_agents(world) if ga.leader]
        members = [ga for ga in self.good_agents(world) if not ga.leader]

        # Agents are rewarded to run away from adveraries
        shape = True
        if shape:
            rew += 0.1 * sum([world.cached_dist_mag[adv.i, agent.i] for adv in adversaries])
        if agent.collide:
            rew -= 5 * sum([self.is_collision(adv, agent, world) for adv in adversaries])

        # Agents are rewarded to collect food and deposit into the leaders
        shape = True
        if not agent.leader:
            if not agent.holding and shape:
                rew -= 0.1 * min([world.cached_dist_mag[f.i, agent.i] for f in world.foods])
            elif shape:
                rew -= 0.1 * min([world.cached_dist_mag[d.i, agent.i] for d in leaders])
        else:
            if shape:  # reward can optionally be shaped
                # penalize by distance to closest relevant holding agent
                dists_to_holding = [world.cached_dist_mag[agent.i, m.i] for m in members if m.holding]
                rew -= 0.1 * min(dists_to_holding) if len(dists_to_holding) > 0 else 0
            for m in members:
                if not m.holding:
                    rew += 5 * sum([self.is_collision(f, m, world) for f in world.foods])
                else:
                    rew += 5 * sum([self.is_collision(l, m, world) for l in leaders])

        # # Agents are rewarded to run away from adveraries
        # if agent.leader:
        #     for agent in self.good_agents(world):
        #         shape = True
        #         if shape:
        #             rew += 0.1 * sum([world.cached_dist_mag[adv.i, agent.i] for adv in adversaries])
        #         if agent.collide:
        #             rew -= 5 * sum([self.is_collision(adv, agent, world) for adv in adversaries])

        #         # Agents are rewarded to collect food and deposit into the leaders
        #         shape = True
        #         if not agent.leader:
        #             if not agent.holding and shape:
        #                 rew -= 0.1 * min([world.cached_dist_mag[f.i, agent.i] for f in world.foods])
        #             elif shape:
        #                 rew -= 0.1 * min([world.cached_dist_mag[d.i, agent.i] for d in leaders])
        #         else:
        #             if shape:  # reward can optionally be shaped
        #                 # penalize by distance to closest relevant holding agent
        #                 dists_to_holding = [world.cached_dist_mag[agent.i, m.i] for m in members if m.holding]
        #                 rew -= 0.1 * min(dists_to_holding) if len(dists_to_holding) > 0 else 0
        #             for m in members:
        #                 if not m.holding:
        #                     rew += 5 * sum([self.is_collision(f, m, world) for f in world.foods])
        #                 else:
        #                     rew += 5 * sum([self.is_collision(l, m, world) for l in leaders])

        return rew

    def adversary_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        shape = True
        if shape:
            rew -= 0.1 * min([world.cached_dist_mag[agent.i, ga.i] for ga in good_agents])
        if agent.collide:
            rew += 5 * sum([self.is_collision(agent, ga, world) for ga in good_agents])

        # The leader receives extra bonus when a good agent holding food is catched
        if agent.leader:
            for ga in good_agents:
                if ga.holding:
                    rew += 10 * sum([self.is_collision(adv, ga, world) for adv in adversaries])

        # if agent.leader:
        #     for agent in adversaries:
        #         shape = True
        #         if shape:
        #             rew -= 0.1 * min([world.cached_dist_mag[agent.i, ga.i] for ga in good_agents])
        #         if agent.collide:
        #             rew += 5 * sum([self.is_collision(agent, ga, world) for ga in good_agents])

        #         # The leader receives extra bonus when a good agent holding food is catched
        #         if agent.leader:
        #             for ga in good_agents:
        #                 if ga.holding:
        #                     rew += 10 * sum([self.is_collision(adv, ga, world) for adv in adversaries])

        return rew

    def observation2(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(world.cached_dist_vect[agent.i, entity.i])

        in_forest = [np.array([-1.]), np.array([-1.])]
        inf1 = False
        inf2 = False
        if self.is_collision(agent, world.forests[0], world):
            in_forest[0] = np.array([1.])
            inf1= True
        if self.is_collision(agent, world.forests[1], world):
            in_forest[1] = np.array([1.])
            inf2 = True

        # other agents
        other_pos = []
        other_vel = []
        other_holding = []
        for other in world.agents:
            if other is agent: continue
            oth_f1 = self.is_collision(other, world.forests[0], world)
            oth_f2 = self.is_collision(other, world.forests[1], world)
            if (inf1 and oth_f1) or (inf2 and oth_f2) or (not inf1 and not oth_f1 and not inf2 and not oth_f2) or agent.leader:
                other_pos.append(world.cached_dist_vect[agent.i, other.i])
                other_vel.append(other.state.p_vel)
                if not other.adversary and not other.leader:
                    if other.holding:
                        other_holding.append(np.array([1.]))
                    else:
                        other_holding.append(np.array([-1.]))
            else:
                other_pos.append(np.array([0., 0.]))
                other_vel.append(np.array([0., 0.]))
                if not other.adversary and not other.leader:
                    other_holding.append(np.array([0.]))

        holding = []
        if not agent.adversary and not agent.leader:
            if agent.holding:
                holding.append(np.array([1.]))
            else:
                holding.append(np.array([-1.]))

        ob = np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + holding + other_pos + other_vel + other_holding + entity_pos + in_forest)

        return ob.astype(np.float32)
