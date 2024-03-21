import random
import math
import numpy as np

from networkx.algorithms import all_pairs_shortest_path_length

from flatland.envs.rail_env import RailEnv
from flatland.graphs.graph_utils import RailEnvGraph

from benchmark.core import Task


class TaskGenerator():

    def __init__(self, pick_ups, drop_offs, env: RailEnv):
        self.pick_ups = pick_ups
        self.drop_offs = drop_offs
        self.env = env
        self.tick_next = 0
        rail_graph = RailEnvGraph(self.env)
        graph = rail_graph.graph_rail_grid()
        self.shortest_path_length = dict(all_pairs_shortest_path_length(graph))

    def generate_task(self) -> Task:
        pass

    def get_tick_next(self) -> int:
        return self.tick_next


class TaskGeneratorAnyFree(TaskGenerator):
    def __init__(self, pick_ups, drop_offs, env: RailEnv, agvs):
        super().__init__(pick_ups, drop_offs, env)
        self.agvs = agvs
        self.tick_of_last_generated = -1

    def generate_task(self) -> Task:
        pick_up = random.sample(self.pick_ups, 1)[0]
        drop_off = random.sample(self.drop_offs, 1)[0]
        while pick_up == drop_off:
            drop_off = random.sample(self.drop_offs, 1)[0]
        current_tick = self.env._elapsed_steps
        # todo: define smart deadline
        t = Task(pick_up, drop_off, current_tick + 20)
        self.tick_of_last_generated = self.env._elapsed_steps
        return t

    def get_tick_next(self) -> int:
        if self.tick_of_last_generated == self.env._elapsed_steps:
            self.tick_next = math.inf
            return self.tick_next
        else:
            for i in range(self.env.number_of_agents):
                # if AGV idle
                if self.agvs[i].task_in_work == None and self.agvs[i].tasks.empty():
                    # if AGV on the TG loop
                    if self.env.agents[i].position in self.shortest_path_length[self.pick_ups[0]].keys():
                        print(
                            f"--TaskGeneratorAnyFree tick_next: {self.env._elapsed_steps}.--")
                        self.tick_next = self.env._elapsed_steps
                        return self.tick_next
        self.tick_next = math.inf
        return self.tick_next


class TaskGeneratorPoisson(TaskGenerator):
    def __init__(self, pick_ups, drop_offs, env: RailEnv, average_tick_next):
        super().__init__(pick_ups, drop_offs, env)
        self.average_tick_next = average_tick_next

    def generate_task(self) -> Task:
        pick_up = random.sample(self.pick_ups, 1)[0]
        drop_off = random.sample(self.drop_offs, 1)[0]
        while pick_up == drop_off:
            drop_off = random.sample(self.drop_offs, 1)[0]
        current_tick = self.env._elapsed_steps
        # todo: define smart deadline
        t = Task(pick_up, drop_off, current_tick + 20)

        # Generate a random number from an exponential distribution
        self.tick_next = self.env._elapsed_steps + \
            int(np.random.exponential(self.average_tick_next))

        return t

    def get_tick_next(self) -> int:
        if self.env._elapsed_steps == 0 and self.tick_next == 0:
            self.tick_next = int(np.random.exponential(self.average_tick_next))

        print(f"--TaskGeneratorPoisson tick_next: {self.tick_next}.--")

        return self.tick_next
