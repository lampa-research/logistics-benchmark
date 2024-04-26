import random

from networkx.algorithms import all_pairs_shortest_path, all_pairs_shortest_path_length

from flatland.envs.rail_env import RailEnv
from flatland.graphs.graph_utils import RailEnvGraph

from benchmark.core import Task


class Dispatcher():

    def __init__(self, env: RailEnv):
        self.env = env

        rail_graph = RailEnvGraph(self.env)
        graph = rail_graph.graph_rail_grid()
        self.shortest_path_length = dict(all_pairs_shortest_path_length(graph))

    def assign(self, t: Task):
        pass


class DispatcherRandom(Dispatcher):

    def assign(self, t: Task):
        agent_indices = list(range(self.env.number_of_agents))

        # keep indices that have corresponding keys in self.shortest_path_length
        agent_indices = [index for index in agent_indices if self.env.agents[index]
                         .position in self.shortest_path_length[t.pick_up].keys()]

        return random.choice(agent_indices)


class DispatcherClosestFree(Dispatcher):
    def __init__(self, env: RailEnv, agvs):
        super().__init__(env)
        self.agvs = agvs

    def assign(self, t: Task):

        agent_indices = list(range(self.env.number_of_agents))

        # keep indices that have corresponding keys in self.shortest_path_length
        agent_indices = [index for index in agent_indices if self.env.agents[index]
                         .position in self.shortest_path_length[t.pick_up].keys()]

        # keep agent indices the corresponding agvs of which have the minimum number of tasks
        task_queue_len = {}
        for i in agent_indices:
            if self.agvs[i].task_in_work is None:
                task_queue_len[i] = len(self.agvs[i].tasks)
            else:
                task_queue_len[i] = len(self.agvs[i].tasks) + 1
        min_queue_len = min(task_queue_len.values())
        agent_indices = [
            index for index, length in task_queue_len.items() if length == min_queue_len]

        # keep agent indices that have the shortest path
        path_len = {}
        for i in agent_indices:
            if self.agvs[i].task_in_work is None and len(self.agvs[i].tasks) == 0:
                path_len[i] = self.shortest_path_length[
                    self.env.agents[i].position][t.pick_up]
            elif self.agvs[i].task_in_work is not None and len(self.agvs[i].tasks) == 0:
                path_len[i] = self.shortest_path_length[
                    self.agvs[i].task_in_work.drop_off][t.pick_up]
            else:
                path_len[i] = self.shortest_path_length[
                    self.agvs[i].last_task_in_queue.drop_off][t.pick_up]
        min_path_len = min(path_len.values())
        agent_indices = [index for index,
                         length in path_len.items() if length == min_path_len]

        closest_free_agv_index = -1
        if len(agent_indices) == 1:
            closest_free_agv_index = agent_indices[0]
        # if there are more agvs with shortest queue length and shortest path from last target to new task, chose randomly
        else:
            closest_free_agv_index = random.choice(agent_indices)

        # print("------")
        # print(f"Task_queue_lengths: {task_queue_len}")
        # print(f"Min task_queue_len: {min_task_queue_len}, min_queue_len indices: {min_queue_agv_indices}")
        # print(f"Path_lengths: {path_len}")
        # print(f"Min path_len: {min_path_len}, min_path_len indices: {min_path_agv_indices}")
        # print(f"Chosen agv: {closest_free_agv_index}")
        # print("------")

        return closest_free_agv_index

    class DispatcherExplicit(Dispatcher):
        def __init__(self, env: RailEnv):
            super().__init__(env)
            self.agent_id = 0

        def set_agent(self, agent_id: int):
            self.agent_id = agent_id

        def assign(self, t: Task):
            return self.agent_id 
