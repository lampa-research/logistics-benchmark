import random
import math
import subprocess
import xml.etree.ElementTree as ET

from flatland.envs.rail_env import RailEnv
from flatland.graphs.graph_utils import RailEnvGraph

from benchmark.core import Task

from libPythonCBS import PythonCBS

import networkx as nx
import matplotlib.pyplot as plt


class Planner():

    def __init__(self, env: RailEnv):
        self.env = env

    def determine_action(self, position, orientation, next_position):
        # Define a dictionary to map orientation to vector changes
        direction_vectors = {
            0: (-1, 0),  # north
            1: (0, 1),   # east
            2: (1, 0),   # south
            3: (0, -1)   # west
        }

        if next_position[0] < 0 or next_position[1] < 0:
            return 4

        # Check if we've reached the end of the plan
        if position == next_position:
            return 4  # Stop

        # Determine the required direction to the next position
        direction_to_next = (
            next_position[0] - position[0], next_position[1] - position[1])

        # Get the current orientation vector
        current_direction_vector = direction_vectors[orientation]

        # Determine if we need to turn left, right or go forward
        if current_direction_vector == direction_to_next:
            return 2  # Go forward
        elif ((current_direction_vector[0] == direction_to_next[1]) and
              (current_direction_vector[1] == -direction_to_next[0])):
            return 1  # Turn right
        elif ((current_direction_vector[0] == -direction_to_next[1]) and
              (current_direction_vector[1] == direction_to_next[0])):
            return 3  # Turn left
        else:
            return 2  # Go forward
        
    def determine_orientation(self, position, orientation, next_position):
        direction_vectors = {
            (-1, 0): 0,  # north
            (0, 1): 1,   # east
            (1, 0): 2,   # south
            (0, -1): 3  # west
        }
        diff = tuple(x - y for x, y in zip(next_position, position)) # element-wise subtract
        if diff in direction_vectors.keys():
            return direction_vectors[diff]
        else:
            return orientation
        
    def plan(self):
        pass

    def get_runtime(self):
        pass

class PlannerCBS(Planner):

    def __init__(self, env: RailEnv):
        super().__init__(env)
        # Parameter initialization
        self.agent_priority_strategy = 3
        self.neighbor_generation_strategy = 3
        self.debug = False
        self.framework = "LNS"
        self.time_limit = 60
        # default settings
        # self.default_group_size = 5
        # self.stop_threshold = 30
        # self.max_iteration = 1000
        # self.agent_percentage = 1.1
        # self.replan = True
        # self.replan_timelimit = 3.0
        self.default_group_size = 2
        self.stop_threshold = 0
        self.max_iteration = 2
        self.agent_percentage = 1.1
        self.replan = True
        self.replan_timelimit = 3.0

    def plan(self):
        self.solver = PythonCBS(self.env,
                                self.framework,
                                self.time_limit,
                                self.default_group_size,
                                self.debug,
                                self.replan,
                                self.stop_threshold,
                                self.agent_priority_strategy,
                                self.neighbor_generation_strategy)
        self.solver.updateAgents(self.env)
        self.success = self.solver.search(self.agent_percentage, self.max_iteration)
        self.solver.buildMCP()
        result = self.solver.getResult()
        # details = self.solver.getResultDetail()
        result_formatted = [
            [(value // self.env.width, value % self.env.width) for value in row] for row in result]
        result_formatted = [row[1:] for row in result_formatted]
        self.solver.clearMCP()
        return result_formatted

    def get_runtime(self):
        details = self.solver.getResultDetail()
        return details['final_runtime']


class PlannerCCBS(Planner):
    def __init__(self, env: RailEnv):
        super().__init__(env)

        self.runtime = 0

        rail_graph = RailEnvGraph(self.env)
        self.graph = rail_graph.graph_rail_grid()

        # filter out nodes and edges
        nodes_to_remove = []
        for n in self.graph.nodes:
            if len(n) != 3:
                nodes_to_remove.append(n)
        for n in nodes_to_remove:
            self.graph.remove_node(n)

        # modify the graph to be compatible with CCBS
        G = nx.DiGraph()
        node_index = 0
        self.posdir2index = {}
        self.index2posdir = {}

        for x, y, dir in self.graph.nodes.keys():
            self.posdir2index[x, y, dir] = node_index
            self.index2posdir[f"n{node_index}"] = (x, y, dir)
            G.add_node(f"n{node_index}", key0=f"{x},{y}")
            node_index += 1
        for (xIn, yIn, dirIn), (xOut, yOut, dirOut) in self.graph.edges.keys():
            indexIn = f"n{self.posdir2index[xIn, yIn, dirIn]}"
            indexOut = f"n{self.posdir2index[xOut, yOut, dirOut]}"
            G.add_edge(indexIn, indexOut, key1=1)

        nx.graphml.write_graphml(G, "roadmap.xml", named_key_ids=True)

        # draw the network
        # pos = {}
        # for n in G.nodes():
        #     pos[n] = self.index2posdir[n]
        # nx.draw_networkx(G)
        # plt.show()

    def plan(self):
        # create a task file from environment state
        root = ET.Element("root")

        # get agent locations and targets
        for agent in self.env.agents:
            start = self.posdir2index[*agent.position, agent.direction]
            valid_orientations = self.env.get_valid_directions_on_grid(*agent.target)
            valid_orientations = [index for index, value in enumerate(valid_orientations) if value == True]
            target = self.posdir2index[*agent.target, random.choice(valid_orientations)]
            agent = ET.SubElement(root, "agent", start_id=str(start), goal_id=str(target))

        # export the XML file
        tree = ET.ElementTree(root)
        tree.write("tasks.xml", xml_declaration=True)

        # run the CCBS exe and wait for the result
        subprocess.check_call(['./CCBS', 'roadmap.xml', 'tasks.xml', 'config.xml'])

        # read and parse the result file
        tree = ET.parse('tasks_log.xml')
        root = tree.getroot()
        agents_paths = []

        # Iterate over each agent
        for agent in root.find('log').findall('agent'):
            agent_path = []
            goal_i = -1
            goal_j = 7

            # Iterate over each section in this agent's path
            for section in agent.find('path').findall('section'):
                start_i = int(section.get('start_i'))
                start_j = int(section.get('start_j'))
                goal_i = int(section.get('goal_i'))
                goal_j = int(section.get('goal_j'))
                duration = int(math.ceil(float(section.get('duration'))))

                # Assuming straight movement, add start position and repeat for duration
                for _ in range(duration):
                    agent_path.append((start_i, start_j))
            agent_path.append((goal_i, goal_j))

            agents_paths.append(agent_path)

        self.runtime = float(root.find('.//summary').get('time'))

        return agents_paths

    def get_runtime(self):
        return self.runtime  
