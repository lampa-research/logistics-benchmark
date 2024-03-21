import numpy as np
import os

from benchmark.simulation import Simulation

filename = os.getcwd() + "/maps/01_plant/01_plant.json"

simulation = Simulation(filename)
simulation.run()

print(f"Average number of tasks executed: {simulation.mean_tasks_executed}, standard deviation: {simulation.std_tasks_executed}")
print(f"Average time for finding MAPF solution: {simulation.mean_planning_time} ms, standard deviation: {simulation.std_planning_time} ms")
print(f"Successful plannings: {simulation.mean_successful_plannings * 100}%, standard deviation: {simulation.std_successful_plannings * 100}%")
print(f"Network metrics. a: {simulation.alpha}, b: {simulation.beta}, g: {simulation.gamma}, ac: {simulation.algebraic_connectivity}")

input("Press Enter to continue...")
