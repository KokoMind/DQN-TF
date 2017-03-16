""" using https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""
import numpy as np
import random

class ReplayMemory:
    def __init__(self, shape, max_size=1000000):
        """
        input : enviroment instanse  , max size of replay memory

        """
        # add to config
        self.batch_size=32
        self.history=4


        self.max_size = max_size
        self.shape = shape
        self.states = np.zeros(([self.max_size] + self.shape[:2]), dtype=np.uint8)
        # self.next_states = np.zeros(([self.max_size] + self.shape[:2]), dtype=np.uint8)
        self.actions = np.zeros(self.max_size, dtype=np.int8)
        self.rewards = np.zeros(self.max_size, dtype=np.int8)
        self.done = np.zeros(self.max_size, dtype=np.bool)
        self.idx = 0
        self.cnt = 0
        self.prestates = np.empty(([self.batch_size] + self.shape), dtype=np.uint8)
        self.poststates = np.empty(([self.batch_size] + self.shape), dtype=np.uint8)
        try :
            self.load()
        except:
            pass

    def push(self, state, next_state, action, reward, done):
        """
        add transition to memory

        """
        self.states[self.idx] = state
        # self.next_states[self.idx] = next_state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.done[self.idx] = done
        self.idx = (self.idx + 1) % self.max_size
        self.cnt = min(self.cnt + 1, self.max_size)

    def getState(self, index):
        index = index % self.cnt
        if index >= self.history - 1:
            return np.transpose(self.states[(index - (self.history - 1)):(index + 1)],(1,2,0))
        else:
            indexes = [(index - i) % self.cnt for i in reversed(range(self.history))]
            return np.transpose(self.states[indexes],(1,2,0))



    def get_batch(self, batch_size=64):
        """
        returns : states,actions,rewards,done
        """
        indexes = []
        while len(indexes) < batch_size:
            while True:
                index = random.randint(self.history, self.cnt - 1)
                if index >= self.idx and index - self.history < self.idx:
                    continue
                if self.done[(index - self.history):index].any():
                    continue
                break

            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.done[indexes]
        return self.prestates,self.poststates, actions, rewards,  terminals

    def save(self):
        np.save('states',self.states)
        # np.save('next_states',self.next_states)
        np.save('actions',self.actions)
        np.save('rewards',self.rewards)
        np.save('done',self.done)


    def load(self):
        self.states=np.load('states_new.npy')
        # self.next_states=np.load('next_states.npy')
        self.actions=np.load('actions_new.npy')
        self.rewards=np.load('rewards_new.npy')
        self.done=np.load('done_new.npy')
        self.idx=499999
        self.cnt=499999