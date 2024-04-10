# Design of observations and rewards

## Observations

Clip everything!!!

### Distance to pick-up

- Distance from the current AGV position to the pick-up location $d(a_i,t_{j,p})$.
- Normalised to maximum distance between two points on the roadmap $\max(d(r_i, r_j))$, where $r_i, r_j \in R$. [0-1]

### Distance to complete all tasks in queue

- $d(a_j,t_{0,p}) + \sum_{i=0}^{N_q} d(t_{i,p},t_{i,d}) + d(t_{i,d},t_{i+1,p}) ...$
- Distance from the agent's current position to the pick-up of the first task in queue + distance to complete all tasks (pick-up to drop-off + drop_off to next pick_up).
- Normalise to 2 * queue length * max distance between two stations on the roadmap
- Normalise Log[distance for tasks + 1] / Log[tole zgoraj]

### Task due from current step

- Normalise to diff
  - min distance between two stations + time_buffer_min -> 0
  + max distance between two stations + time_buffer_max -> 1

### Task distance (pick-up to drop-off)

- Normalize to max distance between two stations

### Minimum of distance for all tasks in AGV queue (all agents except i)

- Calculated from normalised values

### Average of distance for all tasks in AGV queue (all agents except i)

- Calculated from normalised values

## Test environment

## Rewards

Zero reward for tasks completed in time. Penalty for delays, down to -1. Reward is shared, X for 1, 1-X shared.

- Log, Linear, Quadratic: try out

Clip delay to max_delay first

Log[delay + 1]/Log[max_delay + 1]

delay^2/max_delay^2

Q: how to assess max_delay?

