# LogisticsBenchmark

LogisticsBenchmark is a tool for designing and benchmarking roadmaps for Autonomous Guided Vehicles (AGVs) in intralogistic environments. It provides a common platform for researchers and practitioners to create, simulate, and evaluate AGV roadmaps using standardized metrics.

![plant_2a](https://github.com/lampa-research/logistics-benchmark/assets/90378892/94b35531-264d-4ec8-8386-aee54d3679d8)

## Features

- Design roadmaps using a GUI based on the Tiled software
- Simulate and evaluate roadmaps using the Flatland multi-agent path finding (MAPF) simulator, adapted for lifelong planning and execution
- Evaluate roadmap quality using system throughput, computational complexity of planning, success rate, and graph theory metrics
- Extensible architecture to integrate additional algorithms for task generation, releasing, dispatching, and planning


## Prerequisites

Initialize the submodules.

``` text
git submodule init
git submodule update
```

Create and activate a virtual environment (Conda or Miniconda recommended):

```
conda create --name logistics-benchmark python=3.11
conda activate logistics-benchmark
```

Install libstdcxx-ng:

```
conda install -c conda-forge libstdcxx-ng
```

Install `flatland-rl` requirements [https://github.com/flatland-association/flatland-rl](https://github.com/flatland-association/flatland-rl)
:
```
pip install -r flatland-rl/requirements.txt
```

Install other requirements:
```
pip install imageio pytmx scipy
```

Install flatland:
```
pip install -e flatland-rl
```

Install the planners:

Continuous-CBS [https://github.com/PathPlanning/Continuous-CBS](https://github.com/PathPlanning/Continuous-CBS):

```
cd benchmark/planners/Continuous-CBS/
mkdir build
cd build
cmake ..
make
cp CCBS ../../../../
cd ../../../../
```

Flatland-CBS: [https://github.com/Jiaoyang-Li/Flatland](https://github.com/Jiaoyang-Li/Flatland)

```
cd benchmark/planners/Flatland-CBS/Mapf-solver/
mkdir build
cd build
cmake ..
make
cp libPythonCBS.so ../../../../../
cd ../../../../../
```

Tiled: [https://www.mapeditor.org/](https://www.mapeditor.org/)
- Install Tiled
- Add `benchmark.tsx` tileset
- When designing a new map, generate five separate layers:
    - Stations
    - Walls
    - SafeSpaces
    - Roadmap
    - Agents

<img src="https://github.com/lampa-research/logistics-benchmark/assets/90378892/c4b93d44-01ba-4eb6-9fa7-7d17c9d4d7b9" width="480">


## Run simulation

In `run_benchmark.py`, change the path to the *.json* file of the layout you want to evaluate. In *.json* file, set the correct path to the roadmap you want to evaluate, number of simulation steps, number of repetitions, and problem specific characteristics (releaser, task generators, and planner). For now, two different planners can be used:
- `"PlannerCBS"` (Flatland-CBS): doesn't work on directed maps (set `"directed": "False"` and `"turn_in_place": "False"`)
- `"PlannerCCBS"` (Continuous-CBS): works on both directed and undirected maps.

If you set `"visualize" = "True"`, the simulation will be visualized and a .gif will be saved to the layout folder.

## Roadmap evaluation

At the end of simulation, the main quality metrics are provided in terminal:
- throughput (average number of tasks executed),
- computational complexity of planning (average time for finding MAPF solution),
- success rate (percentage of succefull plannings), and
- graph theory metrics: &alpha;, &beta;, &gamma;, and *ac*