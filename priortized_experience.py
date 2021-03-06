import numpy as np
import tensorflow as tf
from priority_queue import IndexedMaxHeap
from experience_replay import ReplayMemory


class PrioritizedExperienceReplay(ReplayMemory):
    """the PRM class"""

    def __init__(self, sess, config):
        super().__init__(config)
        self.sess = sess
        self.config = config
        self.max_size = config.prm_max_size
        self.beta_grad = config.beta_grad
        self.alpha_grad = config.alpha_grad
        self._segments_num = config.batch_size
        self._queue = IndexedMaxHeap(self.max_size)
        self.init_alpha_beta()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.start_learn = config.prm_init_size
        self._set_boundaries()

    def init_alpha_beta(self):
        with tf.variable_scope('alpha'):
            self.alpha_tensor = tf.Variable(self.config.initial_alpha, trainable=False, name='alpha')
            self.alpha_input = tf.placeholder('float32', None, name='alpha_input')
            self.alpha_assign_op = self.alpha_tensor.assign(self.alpha_input)

        with tf.variable_scope('beta'):
            self.beta_tensor = tf.Variable(self.config.initial_beta, trainable=False, name='beta')
            self.beta_input = tf.placeholder('float32', None, name='beta_input')
            self.beta_assign_op = self.beta_tensor.assign(self.beta_input)

    def update_alpha(self):
        self.sess.run(self.alpha_assign_op, {self.alpha_input: self.alpha_tensor.eval(self.sess) + self.alpha_grad})
        self._set_boundaries()

    def update_beta(self):
        self.sess.run(self.beta_assign_op, {self.beta_input: self.beta_tensor.eval(self.sess) + self.beta_grad})

    def _set_boundaries(self):
        self.distributions = []
        partition_size = (np.floor(self.max_size / self._segments_num)).astype(int)
        for n in range(partition_size, self.max_size + 1, partition_size):
            if self.start_learn <= n <= self.max_size:
                distribution = {}
                probs = np.arange(1, n + 1)
                probs = np.power(probs, -self.alpha_tensor.eval(self.sess))
                probs /= probs.sum()
                distribution['probs'] = probs

                cdf = np.cumsum(distribution['probs'])
                boundries = []
                step = 0
                index = 0
                for _ in range(self._segments_num):
                    while cdf[index] < step:
                        index += 1
                    boundries.append(index)
                    step += 1 / self._segments_num
                boundries.append(n)
                distribution['boundries'] = boundries

                self.distributions.append(distribution)

    def push(self, state, next_state, action, reward, done):
        super().push(state, next_state, action, reward, done)
        self._queue.update(self.idx, self._queue.get_max_priority())

    def balance(self):
        self._queue.balance()

    def update_priority(self, indices, deltas):
        for idx, delta in zip(indices, deltas):
            self._queue.update(idx, delta)

    def sample(self):
        batch_indices = []

        dist_index = (np.floor(self.cnt / self.max_size * self._segments_num)).astype(int)
        partition_size = np.floor(self.max_size / self._segments_num)
        partition_max = dist_index * partition_size
        distribution = self.distributions[dist_index]

        for seg in range(self._segments_num):
            seg_size = distribution['boundries'][seg + 1] - distribution['boundries'][seg]
            batch_indices.append(np.random.choice(seg_size) + distribution['boundries'][seg])

        alpha_pow = [distribution['probs'][v] for v in batch_indices]

        batch_weigts = np.power(np.array(alpha_pow) * partition_max, -self.beta_tensor.eval(self.sess))
        batch_weigts /= np.max(batch_weigts)

        return batch_indices, batch_weigts, self.states[batch_indices], self.next_states[batch_indices], self.actions[batch_indices], self.rewards[batch_indices], self.done[batch_indices]
