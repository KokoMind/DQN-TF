import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import os

class Environment(object):
	"""Wrapping the gym environment"""

	def __init__(self, env_name, experiment_dir, state_processor_params, record_video_every=10):
		"""
		state_processor_params = { "resize_shape": (h, w),
					"crop_box": (y1, x1, y2, x2),
					"rgb": False,
					"frames_num": 1 }
		"""
		self.__env = gym.envs.make(env_name)
		self.__monitor_path = os.path.join(experiment_dir, "monitor")

		if not os.path.exists(self.__monitor_path):
			os.makedirs(self.__monitor_path)

		self.__env = wrappers.Monitor(self.__env, self.__monitor_path, resume=True,
							video_callable=lambda count: count % record_video_every == 0)

		with tf.variable_scope("state_processor"):
			h, w, c = self.__env.observation_space.shape
			self.__input_state = tf.placeholder(shape=[h, w, c], dtype=tf.uint8)
			self.__state = self.__input_state
			if 'gray' in state_processor_params and state_processor_params['gray']:
				self.__state = tf.image.rgb_to_grayscale(self.__state)
				self.__gray = True
			else:
				self.__gray = False
			if 'crop_box' in state_processor_params:
				y1, x1, y2, x2 = state_processor_params['crop_box']
				self.__state = tf.image.crop_to_bounding_box(self.__state, y1, x1, y2, x2)
			if 'resize_shape' in state_processor_params:
				h, w = state_processor_params['resize_shape']
				self.__state = tf.image.resize_images(self.__state, [h, w],
										 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			if 'frames_num' in state_processor_params:
				self.__frames_num = state_processor_params['frames_num']
			else:
				self.__frames_num = 1

	def reset(self, sess):
		state = self__.env.reset()
		state = self.__state_processor(sess, state)
		if self.__frames_num > 1:
			state = np.squeeze(state)
			self.__states_stack = np.stack([state] * self.__frames_num, axis=2)
			return self.__states_stack
		else:
			return state

	def step(self, sess, action):
		next_state, reward, done, _ = self.__env.step(action)
		next_state = self.__state_processor(sess, next_state)
		if self.__frames_num > 1:
			next_state = np.concatenate((next_state, self.__states_stack[:,:,:self.__frames_num-1]), axis=2)
			self.__states_stack = next_state
		return next_state, reward, done

	def sample_action(self):
		return np.random.choice(self.__env.action_space.n)

	def submit(self, api_key):
		gym.upload(self.__monitor_path, api_key=api_key)

	def __state_processor(self, sess, state):
		return sess.run(self.__state, feed_dict={self.__input_state: state})