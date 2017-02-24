from priority_queue import IndexedMaxHeap
from experience_replay import ReplayMemory
import math
import numpy as np
class PrioritizedExperienceReplay(ReplayMemory):
    """the PRM class"""

    def __init__(self, config):
        super.__init__()
        self.max_size = config.prm_max_size
        self.start_learn = config.prm_start_learn
        self.partition_num= config.prm_partition_num
        self.alpha= config.prm_alpha
        self._segments_num = config.batch_size
        self._queue = IndexedMaxHeap(self.max_size)

        self._set_boundaries()
    def _set_boundaries(self):
        self.distributions = []
        self.partition_num
        partition_size = math.floor(self.max_size / self.partition_num)
        for n in range(partition_size, self.size + 1, partition_size):
            if self.learn_start <= n <= self.priority_size:
                distribution = {}
                probs =np.arange(1,n+1)
                probs**=-self.alpha
                probs=probs.mean()
                distribution['probs'] = probs

                cdf = np.cumsum(distribution['probs'])
                strata_ends = []
                step = 0
                index = 0
                for s in range(1, self.batch_size + 1):
                    while cdf[index] < step:
                        index += 1
                    strata_ends.append(index)
                    step += 1 / self.batch_size

                distribution['boundries'] = strata_ends

                self.distributions.append(distribution)

    def push(self, state, next_state, action, reward, done):
        super.push(state, next_state, action, reward, done)
        self._queue.update(self._queue.get_max_priority(), self.idx)

    def balance(self):
        self._queue.balance()
        self.set_boundaries()

    def update_priority(self, indices, deltas):
        for idx, delta in zip(indices, deltas):
            self._queue.update(delta, idx)

    def sample(self):
        pass
