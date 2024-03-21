import math
import copy
from enum import IntEnum

from flatland.envs.rail_env import RailEnv

from benchmark.core import Task


class Releaser():
    def __init__(self, env: RailEnv):
        self.env = env
        self.tick_next = 0
        self.tasks = []

    def add_task(self, task: Task):
        self.tasks.append(task)

    def get_tasks(self):
        tasks = copy.deepcopy(self.tasks)
        self.tasks.clear()
        return tasks

    def get_tick_next(self):
        return self.tick_next


class PoolState(IntEnum):
    FILLING = 0
    EMPTYING = 1


class ReleaserPool(Releaser):
    def __init__(self, env: RailEnv, pool_size: int):
        super().__init__(env)
        self.pool_size = pool_size
        self.pool_state = PoolState.FILLING

    def add_task(self, task: Task):
        super().add_task(task)
        if len(self.tasks) == self.pool_size and self.pool_state == PoolState.FILLING:
            self.pool_state = PoolState.EMPTYING

    def get_tasks(self):
        self.pool_state = PoolState.FILLING
        return super().get_tasks()

    def get_tick_next(self):
        if self.pool_state == PoolState.FILLING:
            return math.inf
        elif self.pool_state == PoolState.EMPTYING:
            return self.env._elapsed_steps
