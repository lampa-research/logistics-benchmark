from array import array
import json
import sys
import os
import imageio
import math
import random
import numpy as np

import networkx as nx
from networkx.algorithms import all_pairs_shortest_path, all_pairs_shortest_path_length

from benchmark.core import AGV
from benchmark.converters import TmxConverter
from benchmark.dispatchers import DispatcherRandom, DispatcherClosestFree
from benchmark.releasers import ReleaserPool
from benchmark.planners import PlannerCBS, PlannerCCBS
from benchmark.task_generators import TaskGeneratorAnyFree, TaskGeneratorPoisson
from benchmark.converters import TmxConverter

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.line_generators import BenchmarkLineGen
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.graphs.graph_utils import RailEnvGraph
from flatland.envs.agent_utils import TrainState


class Simulation():

    def __init__(self, filename=None):
        self.number_of_repetitions = 0
        self.dispatcher = None
        self.releaser = None
        self.task_generators = []
        self.planner = None
        self.layout = None
        self.env = None
        self.agvs = []
        self.renderer = None
        self.filename = filename
        self.tasks_executed = 0
        self.average_runtime = 0
        self.graph = None
        self.reduced_graph = None
        self.shortest_paths = {}
        self.shortest_path_lengths = {}
        self.settings = None
        self.recalculate = False
        self.turn_in_place = False
        self.alpha = []
        self.beta = []
        self.gamma = []
        self.algebraic_connectivity = []
        self.frames = []
        self.runtime = 0
        self.number_of_replannings = 0
        self.plan = []
        self.layout = None

        self.mean_tasks_executed = 0
        self.std_tasks_executed = 0
        self.mean_planning_time = 0
        self.std_planning_time = 0
        self.failed_plannings = 0
        self.successful_plannings = 0
        self.mean_successful_plannings = 0
        self.std_successful_plannings = 0

        self.init_from_file(filename)

        return

    def init_from_file(self, filename):
        if filename != None:
            with open(filename, 'r') as f:
                self.settings = json.load(f)
                self.number_of_repetitions = self.settings["number_of_repetitions"]

                self.layout = TmxConverter(
                    os.path.dirname(filename) + "/" + self.settings["map"], self.settings["directed"])
                self.create_env()
                self.env._max_episode_steps = self.settings["number_of_steps"]

                rail_graph = RailEnvGraph(self.env)
                self.graph = rail_graph.graph_rail_grid()
                self.reduced_graph = rail_graph.reduce_simple_paths()

                # filter the graph
                nodes_to_remove = []
                for n in self.reduced_graph.nodes:
                    if len(n) != 2:
                        nodes_to_remove.append(n)
                for n in nodes_to_remove:
                    self.reduced_graph.remove_node(n)
                self.reduced_graph = self.reduced_graph.to_undirected()
                # calculate graph metrics

                for i, c in enumerate(sorted(nx.connected_components(self.reduced_graph), key=len, reverse=True)):
                    subgraph = self.reduced_graph.subgraph(c)
                    e = len(subgraph.edges)
                    v = len(subgraph.nodes)
                    self.alpha.append((e - v + 1)/(2 * v - 5))
                    self.beta.append(e / v)
                    self.gamma.append(e / (3 * (v - 2)))
                    self.algebraic_connectivity.append(
                        nx.algebraic_connectivity(subgraph))

                self.shortest_paths = dict(
                    all_pairs_shortest_path(self.graph))
                self.shortest_path_lengths = dict(
                    all_pairs_shortest_path_length(self.graph))

                if self.settings["dispatcher"]["type"] == "DispatcherRandom":
                    self.dispatcher = DispatcherRandom(self.env)
                elif self.settings["dispatcher"]["type"] == "DispatcherClosestFree":
                    self.dispatcher = DispatcherClosestFree(
                        self.env, self.agvs)

                if self.settings["releaser"]["type"] == "ReleaserPool":
                    pool_size = self.settings["releaser"]["pool_size"]
                    self.releaser = ReleaserPool(self.env, pool_size)

                for tg_desc in self.settings["task_generators"]:
                    if tg_desc["pick_ups"] == "pick_ups":
                        pick_ups = self.layout.pick_ups
                    elif tg_desc["pick_ups"] == "pick_up_drop_offs":
                        pick_ups = self.layout.pick_up_drop_offs
                    else:
                        pick_ups = [tuple(item)
                                    for item in tg_desc["pick_ups"]]

                    if tg_desc["drop_offs"] == "drop_offs":
                        drop_offs = self.layout.drop_offs
                    elif tg_desc["drop_offs"] == "pick_up_drop_offs":
                        drop_offs = self.layout.pick_up_drop_offs
                    else:
                        drop_offs = [tuple(item)
                                     for item in tg_desc["drop_offs"]]

                    if tg_desc["type"] == "TaskGeneratorAnyFree":
                        self.task_generators.append(TaskGeneratorAnyFree(
                            pick_ups, drop_offs, self.env, self.agvs, tg_desc["time_buffer_min"], tg_desc["time_buffer_max"]))
                    elif tg_desc["type"] == "TaskGeneratorPoisson":
                        average_tick_next = tg_desc["average_tick_next"]
                        self.task_generators.append(TaskGeneratorPoisson(
                            pick_ups, drop_offs, self.env, average_tick_next, tg_desc["time_buffer_min"], tg_desc["time_buffer_max"]))
                if self.settings["planner"]["type"] == "PlannerCBS":
                    self.planner = PlannerCBS(self.env)
                elif self.settings["planner"]["type"] == "PlannerCCBS":
                    self.planner = PlannerCCBS(self.env)

                if self.settings["visualize"] == "True":
                    self.renderer = RenderTool(
                        self.env, self.layout, screen_width=self.layout.width*30, screen_height=self.layout.height*30, agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND, directed=self.settings["directed"])
                    # self.renderer.render_env(show=True, show_observations=False, show_predictions=False)

                self.turn_in_place = self.settings["turn_in_place"]

                for agent in self.env.agents:
                    agent.target = agent.position
                    agv = AGV()
                    self.agvs.append(agv)

    def create_env(self):

        self.env = RailEnv(
            width=self.layout.width,
            height=self.layout.height,
            rail_generator=rail_from_grid_transition_map(self.layout.gridmap),
            line_generator=BenchmarkLineGen(self.layout.initial_agent_poses),
            number_of_agents=len(self.layout.initial_agent_poses),
            remove_agents_at_target=False,
        )
        self.env.reset(regenerate_schedule=False, regenerate_rail=False)

        for i, position in enumerate(self.layout.initial_agent_poses.keys()):
            self.env.agents[i].initial_position = position
            self.env.agents[i].position = position
            self.env.agents[i].old_position = position
            self.env.agents[i].initial_direction = int(
                self.layout.initial_agent_poses[position] / 90)
            self.env.agents[i].direction = int(
                self.layout.initial_agent_poses[position] / 90)
            self.env.agents[i].old_direction = int(
                self.layout.initial_agent_poses[position] / 90)
            self.env.agent_positions[position] = i

    def soft_reset(self):
        self.env.reset(regenerate_rail=False, regenerate_schedule=False)
        for i in range(self.env.agent_positions.shape[0]):
            for j in range(self.env.agent_positions.shape[1]):
                self.env.agent_positions[i][j] = -1
        # reset agent positions to initial positions
        for i, position in enumerate(self.layout.initial_agent_poses.keys()):
            self.env.agents[i].initial_position = position
            self.env.agents[i].position = position
            self.env.agents[i].old_position = position
            self.env.agents[i].initial_direction = int(
                self.layout.initial_agent_poses[position] / 90)
            self.env.agents[i].direction = int(
                self.layout.initial_agent_poses[position] / 90)
            self.env.agents[i].old_direction = int(
                self.layout.initial_agent_poses[position] / 90)
            self.env.agent_positions[position[0]][position[1]] = i

        # reset AGVs (clear tasks etc)
        self.agvs.clear()
        for agent in self.env.agents:
            agent.target = agent.position
            agv = AGV()
            self.agvs.append(agv)

        # reset releaser and task generators
        self.releaser.tick_next = 0
        for tg in self.task_generators:
            tg.tick_next = 0

        self.tasks_executed = 0
        self.average_runtime = 0
        self.successful_plannings = 0
        self.failed_plannings = 0

    def check_task_states(self):
        self.recalculate = False
        # if at pick_up, set target to drop_off
        for i in range(len(self.env.agents)):
            if self.agvs[i].task_in_work != None and self.env.agents[i].position == self.agvs[i].task_in_work.pick_up and self.env.agents[i].state == TrainState.DONE:
                self.env.agents[i].target = self.agvs[i].task_in_work.drop_off
                self.env.agents[i].state = TrainState.WAITING
                self.recalculate = True
                print(
                    f"--Agent {i} is at pick up, new target: {self.env.agents[i].target}--")

        # if at drop_off - task completed, set task_in_work to None
        for i in range(len(self.env.agents)):
            if self.agvs[i].task_in_work != None and self.env.agents[i].position == self.agvs[i].task_in_work.drop_off and self.env.agents[i].state == TrainState.DONE:
                self.agvs[i].delays.append(self.env._elapsed_steps - self.agvs[i].task_in_work.deadline)
                self.agvs[i].task_in_work = None
                self.tasks_executed += 1
                print(
                    f"--Agent {i} is at drop off.--")

    def generate_tasks(self):
        # generate tasks
        for tg in self.task_generators:
            while self.env._elapsed_steps == tg.get_tick_next():
                task = tg.generate_task()
                self.releaser.add_task(task)
                print(
                    f"----NEW TASK GENERATED: ({task.pick_up}, {task.drop_off}, {task.deadline})----")

    def dispatch_tasks(self, tasks):
        for task in tasks:
            agent_id = self.dispatcher.assign(task)
            self.agvs[agent_id].assign_task(task)
            print(
                f"----TASK RELEASED AND DISPATCHED: ({task.pick_up}, {task.drop_off}, {task.deadline}) and assigned to agv {agent_id}. Task in work? {self.agvs[agent_id].task_in_work != None}----")

    def step_agents(self):
        for i in range(len(self.env.agents)):
            # if task is completed and task queue is not empty, assign new task
            if self.agvs[i].task_in_work == None and len(self.agvs[i].tasks) > 0:
                self.agvs[i].start_next_task()
                self.env.agents[i].target = self.agvs[i].task_in_work.pick_up
                self.recalculate = True
                print(
                    f"--New task is being assigned to agent {i}. Agent {i}  has new target: {self.env.agents[i].target}--")
                # if agent has no task in work and no task in queue, new target = closest safe_space
            elif len(self.agvs[i].tasks) == 0 and self.agvs[i].task_in_work == None and self.agvs[i].going_to_safe_space == False:
                # find closest safe space
                if len(self.layout.safe_spaces) > 0:
                    distance_to_safe_space = []
                    safe_space_index = -1
                    for safe_space in self.layout.safe_spaces:
                        try:
                            distance_to_safe_space.append(
                                self.shortest_path_lengths[self.env.agents[i].position][safe_space])
                        except KeyError:
                            distance_to_safe_space.append(math.inf)
                    min_dist = min(distance_to_safe_space)
                    if min_dist != math.inf:
                        closest_safe_space_indices = [index for index, value in enumerate(
                            distance_to_safe_space) if value == min_dist]
                        if len(closest_safe_space_indices) == 1:
                            safe_space_index = closest_safe_space_indices[0]
                        else:
                            safe_space_index = random.choice(
                                closest_safe_space_indices)
                        self.env.agents[i].target = self.layout.safe_spaces[safe_space_index]
                        self.agvs[i].going_to_safe_space = True
                        self.recalculate = True
                        print(
                            f"--Agv {i} has no task, going to safe space.--")
                        print(
                            f"Distances to safe spaces: {distance_to_safe_space}, min distance: {min_dist}, closest indices: {closest_safe_space_indices}, chosen index: {safe_space_index}")
                        print("---------")

        #  recalculate if there is new target
        if self.recalculate:
            for i in range(len(self.env.agents)):
                self.env.agents[i].initial_position = self.env.agents[i].position
                self.env.agents[i].initial_direction = self.env.agents[i].direction
                self.env.agents[i].state = TrainState.WAITING
            print()
            print("---recalculating---")
            self.plan = self.planner.plan()
            self.runtime += self.planner.get_runtime()
            self.number_of_replannings += 1

            if self.settings["planner"]["type"] == "PlannerCBS":
                if self.planner.success:
                    self.successful_plannings += 1
                else:
                    self.failed_plannings += 1

            elif self.settings["planner"]["type"] == "PlannerCCBS":
                if self.planner.get_runtime() < 1.0:
                    self.successful_plannings += 1
                else:
                    self.failed_plannings += 1

        # Chose an action for each agent in the environment
        # actions = solver.getActions(env, steps, replan_timelimit) # steps: current timestep
        positions = []
        directions = []
        actions = {}
        for i in range(len(self.env.agents)):
            positions.append(self.env.agents[i].position)
            directions.append(self.env.agents[i].direction)
            self.env.agents[i].state = TrainState.MOVING
            # if self.plan:
            if len(self.plan[i]) > 0:
                actions[i] = self.planner.determine_action(
                    self.env.agents[i].position, self.env.agents[i].direction, self.plan[i][0])
            else:
                actions[i] = 4

        next_obs, all_rewards, done, _ = self.env.step(actions)

        if self.turn_in_place == "True":
            for i in range(len(self.env.agents)):
                if len(self.plan[i]) > 0:
                    if self.plan[i][0][0] != -1:
                        self.env.agents[i].position = self.plan[i][0]
                        self.env.agents[i].direction = self.planner.determine_orientation(
                            positions[i], directions[i], self.plan[i][0])

        self.env.active_agents.clear()

        # Print agent positions and first 5 steps of plan (including current position)
        for i in range(len(self.env.agents)):
            print(
                f"Agent {i} position: {self.env.agents[i].position}, target: {self.env.agents[i].target}, orientation: {self.env.agents[i].direction}, action: {actions[i]}, state: {self.env.agents[i].state}, and plan: {self.plan[i][:10]}")
            if len(self.plan[i]) > 0:
                if self.plan[i][0] != (-1, 7):
                    if self.env.agents[i].position != self.plan[i][0]:
                        pass
        print()
        self.plan = [row[1:] if len(
            row) > 0 else row for row in self.plan]

        if self.renderer is not None:
            self.renderer.render_env(
                show=True, show_observations=False, show_predictions=False)
            rendered_frame = self.renderer.get_image()
            self.frames.append(rendered_frame)

    def save_results(self):
        # Save gif
        if self.renderer is not None:
            imageio.mimsave(os.path.dirname(self.filename) +
                            "/" + "animation.gif", self.frames, duration=0.05)
            print("GIF saved.")
            self.renderer.close_window()

        # get average_runtime
        self.average_runtime = self.runtime / self.number_of_replannings

    def run(self):

        tasks_executed = []
        planning_time = []
        successful_plannings = []

        for _ in range(self.number_of_repetitions):

            self.soft_reset()

            for _ in range(self.env._max_episode_steps):
                print(f"------TICK {self.env._elapsed_steps}------")

                self.check_task_states()
                self.generate_tasks()
                if self.releaser.get_tick_next() == self.env._elapsed_steps:
                    tasks = self.releaser.get_tasks()
                    self.dispatch_tasks(tasks)
                    tasks.clear()
                self.step_agents()

            self.save_results()
            tasks_executed.append(self.tasks_executed)
            planning_time.append(self.average_runtime)
            successful_plannings.append(self.successful_plannings / (self.successful_plannings + self.failed_plannings))

        self.mean_tasks_executed = np.mean(tasks_executed)
        self.std_tasks_executed = np.std(tasks_executed)
        self.mean_planning_time = np.mean(planning_time)
        self.std_planning_time = np.std(planning_time)
        self.mean_successful_plannings = np.mean(successful_plannings)
        self.std_successful_plannings = np.std(successful_plannings)
