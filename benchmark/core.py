import queue


class Task():

    def __init__(self, pick_up, drop_off, deadline):
        self.pick_up = pick_up
        self.drop_off = drop_off
        self.deadline = deadline

    def __str__(self):
        return f'From: {self.pick_up} To: {self.drop_off} Until: {self.deadline}'


class AGV():

    def __init__(self):
        self.tasks = queue.Queue()
        self.task_in_work = None  # Task((0,0), (0,0), 0)
        self.deadline = 0  # deadline for the task being executed
        self.going_to_safe_space = False
        self.last_task_in_queue = None
        self.delays = [] # a list of delays for completed tasks

    def assign_task(self, t: Task):
        self.tasks.put(t)
        self.last_task_in_queue = t

    def start_next_task(self):
        if not self.tasks.empty():
            t = self.tasks.get(0)
            self.task_in_work = t
            self.deadline = t.deadline
            self.going_to_safe_space = False
            if self.tasks.empty():
                self.last_task_in_queue = None
            return t
        else:
            return None
