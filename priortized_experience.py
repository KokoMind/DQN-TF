from priority_queue import IndexedMaxHeap
from experience_replay import ReplayMemory


class PrioritizedExperienceReplay(ReplayMemory):
    """the PRM class"""

    def __init__(self, config):
        super.__init__()
        self.max_size = config.prm_max_size
        self.segments_num = config.batch_size
        self.queue = IndexedMaxHeap(self.max_size)

    def set_boundaries(self):
        pass

    def push(self, state, next_state, action, reward, done):
        super.push(state, next_state, action, reward, done)
        self.queue.update(self.queue.get_max_priority(), self.idx)

    def balance(self):
        self.queue.balance()
        self.set_boundaries()

    def update_priority(self, indices, deltas):
        for idx, delta in zip(indices, deltas):
            self.queue.update(delta, idx)

    def sample(self):
        pass
