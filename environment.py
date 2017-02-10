import gym
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
		self.env = gym.envs.make(env_name)
		self.monitor_path = os.path.join(experiment_dir, "monitor")

     	if not os.path.exists(self.monitor_path):
        	os.makedirs(self.monitor_path)

	    env = wrappers.Monitor(env, self.monitor_path, resume=True,
                           video_callable=lambda count: count % record_video_every == 0)

		with tf.variable_scope("state_processor"):
			h, w, c = self.env.observation_space.shape
			self.state = tf.placeholder(shape=[h, w, c], dtype=tf.uint8)
			if 'rgb' in state_processor_params and state_processor_params['rgb']:
				self.state = tf.image.rgb_to_grayscale(self.state)
				self.rgb = True
			else 
				self.rgb = False
			if 'crop_box' in state_processor_params:
				y1, x1, y2, x2 = state_processor_params['crop_box']
				self.state = tf.image.crop_to_bounding_box(self.state, y1, x1, y2, x2)
			if 'resize_shape' in state_processor_params:
				h, w = state_processor_params['resize_images']
				self.state = tf.image.resize_images(self.state, [h, w],
										 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			if 'frames_num' in state_processor_params:
				self.frames_num = state_processor_params['frames_num']
			else
				self.frames_num = 1

	def reset(self, sess):
		state = self.env.reset()
		state = _state_processor(sess, state)
		if self.frames_num > 1:
			self.states_stack = np.stack([state] * self.frames_num, axis=2)
			return self.states_stack
		elif not self.rgb:
			state = np.expand_dims(state, 2)
			return state

	def step(self, sess, action):
		next_state, reward, done, _ = self.env.step(action)
		next_state = _state_processor(sess, next_state)
		if not self.rgb:
			next_state = np.expand_dims(next_state, 2)
		if self.frames_num > 1:
			next_state = np.concatenate((next_state, state[:,:,:self.frames_num-1]), axis=2)
			self.states_stack = next_state
		return next_state, reward, done

	def submit(self, api_key):
		gym.upload(self.monitor_path, api_key=api_key)

	def _state_processor(self, sess, state):
		return sess.run(self.state, feed_dict={self.state: state})