import pytmx
import random
import xml.etree.ElementTree as ET
from itertools import product
from flatland.core.transition_map import GridTransitionMap
from flatland.core.grid.rail_env_grid import RailEnvTransitions

from flatland.core.grid.grid4_utils import get_new_position


class TmxConverter():

    def __init__(self, filename: str, directed: str):
        tmx_data = pytmx.TiledMap(filename)
        tree = ET.parse(filename)
        root = tree.getroot()
        self.width = tmx_data.width
        self.height = tmx_data.height

        layer_names = ["Roadmap", "Stations", "Walls",
                       "SafeSpaces", "Agents"]
        tile_matrices = []

        for layer_name in layer_names:
            layer_data_element = root.find(
                f".//layer[@name='{layer_name}']/data")
            if layer_data_element is not None:
                layer_csv_data = layer_data_element.text.strip()
                layer_tile_ids = [int(tile_id)
                                  for tile_id in layer_csv_data.split(",")]
            else:
                # Handle the case when the layer doesn't exist
                layer_tile_ids = [0] * (self.width * self.height)

            tile_matrix = []
            for y in range(self.height):
                row = layer_tile_ids[y * self.width: (y + 1) * self.width]
                tile_matrix.append(row)

            tile_matrices.append(tile_matrix)

        extracted_values = [[self.extract_tile_info(
            number)[::-1] for number in row] for row in tile_matrices[0]] #roadmap

        # Access map properties
        print("Map width:", self.width)
        print("Map height:", self.height)

        # Create a map of tuples, all initialized as (0, 0)
        self.specs = [[(0, 0) for _ in range(self.width)]
                      for _ in range(self.height)]

        # Populate specs using extracted_values
        for y in range(tmx_data.height):
            for x in range(tmx_data.width):
                tile_value = extracted_values[y][x]
                if tile_value != (0, 0):
                    # Unpack the tile information from tile_value
                    tile_ID, rotation_code = tile_value
                    # Populate specs with adjusted Tile ID and rotation angle based on the Rotation Code
                    # dictionary lookup
                    self.specs[y][x] = (
                        tile_ID - 1, {0: 0, 10: 90, 12: 180, 6: 270}.get(rotation_code, 0))

        transitions = RailEnvTransitions()
        self.gridmap = GridTransitionMap(
            width=self.width, height=self.height, transitions=transitions)
        self.gridmap.grid.fill(0)

        # constraints (placed rails)
        for row, row_values in enumerate(self.specs):
            for column, (tile_ID, rotation_code) in enumerate(row_values):
                if tile_ID != 0:
                    effective_transition_cell = transitions.rotate_transition(
                        transitions.transitions[tile_ID], rotation_code)
                    self.gridmap.set_transitions(
                        (row, column), effective_transition_cell)
                    self.gridmap.grid[row][column] = effective_transition_cell

        if directed == "True":
            # check consistency
            change = True
            while change == True:
                change = False
                for i in range(self.height):
                    for j in range(self.width):
                        cell = self.gridmap.grid[i][j]
                        if i > 0:
                            cell = self.check_consistency('N', cell, self.gridmap.grid[i-1][j])
                        if i < self.height - 1:
                            cell = self.check_consistency('S', cell, self.gridmap.grid[i+1][j])
                        if j > 0:
                            cell = self.check_consistency('W', cell, self.gridmap.grid[i][j-1])
                        if j < self.width - 1:
                            cell = self.check_consistency('E', cell, self.gridmap.grid[i][j+1])
                        if cell != self.gridmap.grid[i][j]:
                            self.gridmap.grid[i][j] = cell
                            self.gridmap.set_transitions((i, j), cell)
                            change = True
                            print(f'Cell {i},{j}: {bin(cell)[2:].zfill(16)} modified due to neighbours: {bin(cell)[2:].zfill(16)}')
        elif directed == "False":
            for i in range(self.height):
                for j in range(self.width):
                    cell = self.gridmap.grid[i][j]
                    cell = self.create_undirected(cell)
                    self.gridmap.grid[i][j] = cell
                    self.gridmap.set_transitions((i, j), cell)

        # Read target and obstacle positions from map data:
        self.stations = []
        self.pick_ups = []
        self.drop_offs = []
        self.pick_up_drop_offs = []
        self.walls = []
        self.safe_spaces = []
        self.initial_agent_poses = {}

        for x in range(self.height):
            for y in range(self.width):
                _, tile_ID1 = self.extract_tile_info(tile_matrices[1][x][y]) # stations layer
                _, tile_ID2 = self.extract_tile_info(tile_matrices[2][x][y]) # walls layer
                _, tile_ID3 = self.extract_tile_info(tile_matrices[3][x][y]) # safe_spaces layer
                initial_agent_orientation, tile_ID4 = self.extract_tile_info( 
                    tile_matrices[4][x][y]) # agents layer

                # ignore empty tiles (case 0)
                if tile_matrices[1][x][y] != 0 and tile_ID1 - 1 != 0:
                    self.stations.append((x, y))
                    if tile_ID1 - 1 == 27:
                        self.pick_up_drop_offs.append((x, y))
                    elif tile_ID1 - 1 == 28:
                        self.pick_ups.append((x, y))
                    elif tile_ID1 - 1 == 29:
                        self.drop_offs.append((x, y))

                # ignore empty tiles (case 0)
                if tile_matrices[2][x][y] != 0 and tile_ID2 - 1 != 0:
                    self.walls.append((x, y))

                # ignore empty tiles (case 0)
                if tile_matrices[3][x][y] != 0 and tile_ID3 - 1 != 0:
                    self.safe_spaces.append((x, y))

                # # ignore empty tiles (case 0)
                if tile_matrices[4][x][y] != 0 and tile_ID4 - 1 != 0:
                    self.initial_agent_poses[(x, y)] = {0: 0, 10: 90, 12: 180, 6: 270}[
                        initial_agent_orientation]

        if not self.stations:
            self.stations = None

        if not self.walls:
            self.walls = None

        return

    # Function to extract Rotation Code and Tile ID from a number
    def extract_tile_info(self, number):
        # Convert the number to a 32-bit binary representation
        binary_value = format(number, '032b')
        # Extract the first 4 bits
        rotation_code = int(binary_value[:4], 2)
        tile_ID = int(binary_value[-16:], 2)       # Extract the last 4 bits
        return rotation_code, tile_ID

    def check_consistency(self, direction, cell, neighbour):
        cell_in_masks = {
            'N': 0b0000000011110000,
            'E': 0b0000000000001111,
            'S': 0b1111000000000000,
            'W': 0b0000111100000000,
        }

        cell_out_masks = {
            'N': 0b1000100010001000,
            'E': 0b0100010001000100,
            'S': 0b0010001000100010,
            'W': 0b0001000100010001,
        }

        neighbour_in_masks = {
            'N': 0b1111000000000000,
            'E': 0b0000111100000000,
            'S': 0b0000000011110000,
            'W': 0b0000000000001111,
        }

        neighbour_out_masks = {
            'N': 0b0010001000100010,
            'E': 0b0001000100010001,
            'S': 0b1000100010001000,
            'W': 0b0100010001000100,
        }

        modified_cell = cell
        if neighbour & neighbour_out_masks[direction] == 0:
            # print(f'Modifying due to neighbour out: {bin(modified_cell)[2:].zfill(16)}')
            modified_cell = modified_cell & ~cell_in_masks[direction]
            # print(f'Modified due to neighbour out: {bin(modified_cell)[2:].zfill(16)}')
        if neighbour & neighbour_in_masks[direction] == 0:
            # print(f'Modifying due to neighbour in: {bin(modified_cell)[2:].zfill(16)}')
            modified_cell = modified_cell & ~cell_out_masks[direction]
            # print(f'Modified due to neighbour out: {bin(modified_cell)[2:].zfill(16)}')
        return modified_cell
    
    def create_undirected(self, cell):
        if cell == 0b1000000000000000 or cell == 0b0000000000100000:
            return 0b1000000000100000
        elif cell == 0b0000010000000000 or cell == 0b0000000000000001:
            return 0b0000010000000001
        else:
            return cell
