import numpy as np


class ReplayMemory(object):
    def __init__(self, config):
        """
        input : enviroment instanse  , max size of replay memory

        """
        self.max_size = config.rep_max_size
        self.shape = config.state_shape
        self.states = np.zeros(([self.max_size] + self.shape), dtype=np.uint8)
        self.next_states = np.zeros(([self.max_size] + self.shape), dtype=np.uint8)
        self.actions = np.zeros(self.max_size, dtype=np.int8)
        self.rewards = np.zeros(self.max_size, dtype=np.int8)
        self.done = np.zeros(self.max_size, dtype=np.bool)
        self.idx = 0
        self.cnt = 0

    def push(self, state, next_state, action, reward, done):
        """
        add transition to memory

        """
        assert state.shape == tuple(self.shape)
        self.states[self.idx] = state
        self.next_states[self.idx] = next_state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.done[self.idx] = done
        self.idx = (self.idx + 1) % self.max_size
        self.cnt = min(self.cnt + 1, self.max_size)

    def get_batch(self, batch_size=64):
        """
        returns : states,actions,rewards,done
        """
        idxs = np.random.choice(range(self.cnt), batch_size)
        return self.states[idxs], self.next_states[idxs], self.actions[idxs], self.rewards[idxs], self.done[idxs]
