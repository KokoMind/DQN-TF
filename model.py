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

from layers import conv2d,linear
__version__ = 0.1


class BaseModel(object):
    """A Base Class for any model to be constructed"""

    def __init__(self):
        pass

    def build_model(self,name,trainable,num_outputs,shape):
        self.behaviour_weights = {}
        self.target_weights = {}
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        with tf.variable_scope('behaviour'):
            self.b_x = tf.placeholder(tf.uint8, shape=[None] + shape, name="states")
            self.b_conv1, self.behaviour_weights['conv1_w'], self.behaviour_weights['conv1_b'] = conv2d(self.x,
                                        32, [8, 8], [4, 4], initializer, activation_fn, name='conv1')
            self.b_conv2, self.behaviour_weights['conv2_w'], self.behaviour_weights['conv2_b'] = conv2d(self.conv1,
                                        64, [4, 4], [2, 2], initializer, activation_fn, name='conv2')
            self.b_conv3, self.behaviour_weights['conv3_w'], self.behaviour_weights['conv3_b'] = conv2d(self.conv2,
                                        64, [3, 3], [1, 1], initializer, activation_fn, name='conv3')
            self.b_conv3_flat=tf.contrib.layers.flatten(self.conv3)

            self.b_fc1, self.behaviour_weights['fc1_w'], self.behaviour_weights['fc1_b'] = linear(self.conv3_flat, 512,
                                                                                    activation_fn=activation_fn, name='b_fc1')
            self.b_out, self.behaviour_weights['out_w'], self.behaviour_weights['out_b'] = linear(self.fc1, num_outputs, name='b_out')

        with tf.variable_scope('target'):
            self.t_x = tf.placeholder(tf.uint8, shape=[None] + shape, name="states")
            self.t_conv1, self.target_weights['conv1_w'], self.target_weights['conv1_b'] = conv2d(self.x,
                                                                      32, [8, 8], [4, 4], initializer, activation_fn, name='conv1')
            self.t_conv2, self.target_weights['conv2_w'], self.target_weights['conv2_b'] = conv2d(self.conv1,
                                                                      64, [4, 4], [2, 2], initializer, activation_fn,
                                                                      name='conv2')
            self.t_conv3, self.target_weights['conv3_w'], self.target_weights['conv3_b'] = conv2d(self.conv2,
                                                                      64, [3, 3], [1, 1], initializer, activation_fn,
                                                                      name='conv3')
            self.t_conv3_flat = tf.contrib.layers.flatten(self.conv3)
            self.t_fc1, self.target_weights['fc1_w'], self.target_weights['fc1_b'] = linear(self.conv3_flat, 512, activation_fn=activation_fn,
                                                                                          name='t_fc1')
            self.t_out, self.target_weights['out_w'], self.target_weights['out_b'] = linear(self.fc1, num_outputs, name='t_out')

        with tf.variable_scope('copy'):
            self.copy_from = {}
            self.copy_to = {}

            for name in self.behaviour_weights.keys():
                self.copy_from[name] = tf.placeholder('float32', self.target_weights[name].get_shape().as_list(), name=name)
                self.copy_to[name] = self.target_weights[name].assign(self.copy_from[name])

    def predict(self, sess, state):
        raise NotImplemented()

    def update(self, sess, state, action, y):
        raise NotImplemented()


class DQN(BaseModel):
    """Our Estimator Network"""

    def __init__(self,name="blablabla",sess=None,shape=None,num_action_space=None,dir="./",learning_rate=.0002):
        BaseModel.__init__(self,shape)
        self.name=name
        self.sess = sess
        self.dir=dir
        self.num_actions=num_action_space
        self.shape=shape
        self.learning_rate=learning_rate
        self.saver = tf.train.Saver()

        with tf.variable_scope("DQN"):
            #placeholders
            self.actions = tf.placeholder(tf.float32, shape=[None])
            self.targets = tf.placeholder(tf.float32, [None])
            gather_indices = tf.range(tf.shape(self.b_out)[0]) * tf.shape(self.b_out)[1] + self.actions
            self.action_predictions = tf.gather(tf.reshape(self.b_out, [-1]), gather_indices)

            #loss
            self.losses = tf.squared_difference(self.targets, self.action_predictions)
            self.loss = tf.reduce_mean(self.losses)
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1e-6)
            self.update_step = self.optimizer.minimize(self.loss)

    def update_target_network(self):
        for name in self.behaviour_weights.keys():
            self.copy_to[name].eval({self.copy_from[name]: self.behaviour_weights[name].eval()})

    def predict(self, state,type="behaviour"):

        if type=="behaviour":
            return self.sess.run(self.b_out, { self.b_x: state })
        elif type=="target":
            return self.sess.run(self.t_out, { self.t_x: state })

    def update(self, sess, s, a, y):

        feed_dict = {self.b_x: s, self.targets: y, self.actions: a}
        _,loss = sess.run(
            [self.update_step, self.loss],
            feed_dict)
        return loss


