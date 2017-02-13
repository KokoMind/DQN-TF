import os
import sys
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import itertools

from model import DQN
from experience_replay import ReplayMemory

__version__ = 0.1


class Agent:
    """Our Wasted Agent :P """

    def __init__(self, sess, config, environment):
        # Get the session, config, environment, and create a replaymemory
        self.sess = sess
        self.config = config
        self.environment = environment
        self.memory = ReplayMemory(config.state_shape, config.rep_max_size)

        self.init_dirs()
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)

        self.init_global_step()

        # Intialize the DQN graph which contain 2 Networks Target and Q
        self.estimator = DQN(sess, config, self.environment.n_actions)

        # To initialize all variables
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.saver = tf.train.Saver(max_to_keep=10)

        if config.is_train and config.initial_training:
            pass
        elif config.is_train and config.cont_training:
            self.load()
        elif config.is_play:
            self.load()
        else:
            raise Exception("Please Set proper mode for training or playing")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)

    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir, self.global_step_tensor)

    def init_dirs(self):
        # Create directories for checkpoints and summaries
        self.checkpoint_dir = os.path.join(self.config.experiment_dir, "checkpoints/")
        self.summary_dir = os.path.join(self.config.experiment_dir, "summaries/")

    def init_global_step(self):
        """Create a global step variable to be a reference to the number of iterations"""
        with tf.variable_scope('step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    def init_replay_memory(self):
        # Populate the replay memory with initial experience
        print("initializing replay memory...")

        state = self.environment.reset()
        for i in itertools.count():
            action = self.take_action(state)
            next_state, reward, done = self.observe_and_save(state, self.environment.valid_actions[action])
            if done:
                if i >= self.config.replay_memory_init_size:
                    break
                else:
                    state = self.environment.reset()
            else:
                state = next_state

    def policy_fn(self, fn_type, estimator, n_actions):
        """Function that contain definitions to various number of policy functions and choose between them"""

        def epsilon_greedy(sess, observation, epsilon):
            actions = np.ones(n_actions, dtype=float) * epsilon / n_actions
            q_values = estimator.predict(np.expand_dims(observation, 0))[0]
            best_action = np.argmax(q_values)
            actions[best_action] += (1.0 - epsilon)
            return actions

        def greedy(sess, observation):
            q_values = estimator.predict(np.expand_dims(observation, 0))[0]
            best_action = np.argmax(q_values)
            return best_action

        if fn_type == 'epsilon_greedy':
            return epsilon_greedy
        elif fn_type == 'greedy':
            return greedy
        else:
            raise Exception("Please Select a proper policy function")

    def take_action(self, state):
        """Take the action based on the policy function"""
        action_probs = self.policy(self.sess, state, self.epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def observe_and_save(self, state, action):
        """Function that observe the new state , reward and save it in the memory"""
        next_state, reward, done = self.environment.step(action)
        self.memory.push(state, next_state, action, reward, done)
        return next_state, reward, done

    def update_target_network(self):
        """Update Target network By copying paramter between the two networks in DQN"""
        self.estimator.update_target_network()

    def train_episodic(self):
        """Train the agent in episodic techniques"""

        # Initialize the epsilon, it's step, the policy function, the replay memory
        self.epsilon = self.config.initial_epsilon
        self.epsilon_step = (self.config.initial_epsilon - self.config.final_epsilon) / self.config.exploration_steps
        self.policy = self.policy_fn(self.config.policy_fn, self.estimator, self.environment.n_actions)
        self.init_replay_memory()

        for cur_episode in range(self.config.num_episodes):

            # Save the current checkpoint
            self.save()

            state = self.environment.reset()
            total_reward = 0

            # Take steps in the environment untill terminal state of epsiode
            for t in itertools.count():

                # Update the Global step
                self.global_step_assign_op.eval(session=self.sess, feed_dict={self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})

                # time to update the target estimator
                if self.global_step_tensor.eval(self.sess) % self.config.update_target_estimator_every == 0:
                    self.update_target_network()

                # Calculate the Epsilon for this time step
                # Take an action ..Then observe and save
                self.epsilon = max(self.config.final_epsilon, self.epsilon - self.epsilon_step)
                action = self.take_action(state)
                next_state, reward, done = self.observe_and_save(state, self.environment.valid_actions[action])

                # Sample a minibatch from the replay memory
                state_batch, next_state_batch, action_batch, reward_batch, done_batch = self.memory.get_batch(self.config.batch_size)

                # Calculate targets Then Compute the loss
                q_values_next = self.estimator.predict(next_state_batch, type="target")
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.config.discount_factor * np.amax(q_values_next, axis=1)
                _ = self.estimator.update(state_batch, action_batch, targets_batch)

                total_reward += reward

                if done:  # IF terminal state so exit the episode
                    # Add summaries to tensorboard
                    episode_summary = tf.Summary()
                    episode_summary.value.add(simple_value=total_reward, node_name="episode_reward", tag="episode_reward")
                    episode_summary.value.add(simple_value=t, node_name="episode_length", tag="episode_length")
                    episode_summary.value.add(simple_value=self.epsilon, node_name="epsilon", tag="epsilon")
                    self.summary_writer.add_summary(episode_summary, self.global_step_tensor.eval(self.sess))
                    self.summary_writer.flush()
                    break

                state = next_state

        print("Training Finished")

    def train_continous(self):
        # TODO implement on global step only
        pass

    def play(self, n_episode=10):
        """Function that play greedily on the policy learnt"""
        self.policy = self.policy_fn('greedy', self.estimator, self.environment.n_actions)

        for episode in range(n_episode):

            state = self.environment.reset()

            for t in itertools.count():

                best_action = self.policy(self.sess, state)
                next_state, reward, done = self.environment.step(best_action)

                if done:
                    break

                state = next_state
