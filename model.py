import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import utils
import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from layers import conv2d, linear

__version__ = 0.1


class BaseModel(object):
    """A Base Class for any model to be constructed"""

    def __init__(self):
        pass

    def build_model(self, num_outputs, shape):
        self.behaviour_weights = {}
        self.target_weights = {}
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        with tf.variable_scope('behaviour'):
            self.b_x = tf.placeholder(tf.uint8, shape=[None] + shape, name="states")
            self.b_x = tf.to_float(self.b_x) / 255.0
            self.b_conv1, self.behaviour_weights['conv1_w'], self.behaviour_weights['conv1_b'] = conv2d(self.b_x, 32, [8, 8], [4, 4], initializer, activation_fn,
                                                                                                        name='b_conv1')
            self.b_conv2, self.behaviour_weights['conv2_w'], self.behaviour_weights['conv2_b'] = conv2d(self.b_conv1, 64, [4, 4], [2, 2], initializer, activation_fn,
                                                                                                        name='b_conv2')
            self.b_conv3, self.behaviour_weights['conv3_w'], self.behaviour_weights['conv3_b'] = conv2d(self.b_conv2, 64, [3, 3], [1, 1], initializer, activation_fn,
                                                                                                        name='b_conv3')
            self.b_conv3_flat = tf.contrib.layers.flatten(self.b_conv3)
            self.b_fc1, self.behaviour_weights['fc1_w'], self.behaviour_weights['fc1_b'] = linear(self.b_conv3_flat, 512, activation_fn=activation_fn, name='b_fc1')
            self.b_out, self.behaviour_weights['out_w'], self.behaviour_weights['out_b'] = linear(self.b_fc1, num_outputs, name='b_out')

        with tf.variable_scope('target'):
            self.t_x = tf.placeholder(tf.uint8, shape=[None] + shape, name="states")
            self.t_x = tf.to_float(self.t_x) / 255.0
            self.t_conv1, self.target_weights['conv1_w'], self.target_weights['conv1_b'] = conv2d(self.t_x, 32, [8, 8], [4, 4], initializer, activation_fn,
                                                                                                  name='t_conv1')
            self.t_conv2, self.target_weights['conv2_w'], self.target_weights['conv2_b'] = conv2d(self.t_conv1, 64, [4, 4], [2, 2], initializer, activation_fn,
                                                                                                  name='t_conv2')
            self.t_conv3, self.target_weights['conv3_w'], self.target_weights['conv3_b'] = conv2d(self.t_conv2, 64, [3, 3], [1, 1], initializer, activation_fn,
                                                                                                  name='t_conv3')
            self.t_conv3_flat = tf.contrib.layers.flatten(self.t_conv3)
            self.t_fc1, self.target_weights['fc1_w'], self.target_weights['fc1_b'] = linear(self.t_conv3_flat, 512, activation_fn=activation_fn, name='t_fc1')
            self.t_out, self.target_weights['out_w'], self.target_weights['out_b'] = linear(self.t_fc1, num_outputs, name='t_out')

        with tf.variable_scope('copy'):
            self.copy_from = {}
            self.copy_to = {}
            for name in self.behaviour_weights.keys():
                self.copy_from[name] = tf.placeholder('float32', self.target_weights[name].get_shape().as_list(), name=name)
                self.copy_to[name] = self.target_weights[name].assign(self.copy_from[name])

    def predict(self, state, type):
        raise NotImplemented()

    def update(self, s, a, y):
        raise NotImplemented()


class DQN(BaseModel):
    """   Our Estimator Network   """

    def __init__(self, sess, config, num_actions):
        BaseModel.__init__(self)
        self.name = config.name
        self.sess = sess
        self.config = config
        self.num_actions = num_actions
        self.shape = config.state_shape
        self.learning_rate = config.learning_rate
        self.build_model(num_outputs=self.num_actions, shape=self.shape)
        with tf.variable_scope("DQN"):
            # placeholders
            with tf.name_scope('lose'):
                self.actions = tf.placeholder(tf.int32, shape=[None], name='actions')
                self.targets = tf.placeholder(tf.float32, [None], name='targets')
                self.weights = tf.placeholder(tf.float32, [None], name='weights')
                gather_indices = tf.range(config.batch_size) * tf.shape(self.b_out)[1] + self.actions
                self.action_predictions = tf.gather(tf.reshape(self.b_out, [-1]), gather_indices)
                # loss
                if config.prm:
                    self.losses = tf.mul(tf.squared_difference(self.targets, self.action_predictions), self.weights)
                else:
                    self.losses = tf.squared_difference(self.targets, self.action_predictions)

                self.loss = tf.reduce_mean(self.losses)

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1e-6)
            self.update_step = self.optimizer.minimize(self.loss)

    def update_target_network(self):
        for name in self.behaviour_weights.keys():
            self.sess.run(self.copy_to[name], {self.copy_from[name]: self.behaviour_weights[name].eval(self.sess)})

    def predict(self, states, type="behaviour"):
        if type == "behaviour":
            return self.sess.run(self.b_out, {self.b_x: states})
        elif type == "target":
            return self.sess.run(self.t_out, {self.t_x: states})

    def update(self, s, a, y, weights=None):
        if self.config.prm:
            feed_dict = {self.b_x: s, self.targets: y, self.actions: a, self.weights: weights}
        else:
            feed_dict = {self.b_x: s, self.targets: y, self.actions: a}

        _, loss = self.sess.run([self.update_step, self.loss], feed_dict)
        return loss
