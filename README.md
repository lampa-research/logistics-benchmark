# LogisticsBenchmark

LogisticsBenchmark is a tool for designing and benchmarking roadmaps for Autonomous Guided Vehicles (AGVs) in intralogistic environments. It provides a common platform for researchers and practitioners to create, simulate, and evaluate AGV roadmaps using standardized metrics.

## Features

- Design roadmaps using a GUI based on the Tiled software
- Simulate and evaluate roadmaps using the Flatland multi-agent path finding (MAPF) simulator, adapted for lifelong planning and execution
- Evaluate roadmap quality using system throughput, computational complexity of planning, failure rate, and graph theory metrics
- Extensible architecture to integrate additional algorithms for task generation, releasing, dispatching, and planning

## Prerequisites

Initialize the submodules.

``` text
git submodule init
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

Install Tiled: [https://www.mapeditor.org/](https://www.mapeditor.org/)
