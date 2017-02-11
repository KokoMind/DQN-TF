import os
import sys
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import itertools

from model import Estimator
from experience_replay import ReplayMemory

__author__ = "Mo'men"
__version__ = 0.1


class Agent:
    """Our Wasted Agent :P """

    def __init__(self, sess, config, environment):
        self.sess = sess
        self.config = config
        self.environment = environment
        self.memory = ReplayMemory(config.state_shape, config.rep_mem_max)

        self.init_dirs()

        self.q_predictor = Estimator()
        self.target_estimator = Estimator()

        self.saver = tf.train.Saver()

        self.summary_writer = tf.train.SummaryWriter(self.summary_dir)

        self.init_global_step()

        if config.initial_training:
            pass
        elif config.load_checkpoint:
            self.load()
        else:
            raise Exception("Please Set the mode of the training if initial or loading a checkpoint")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)

    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir)

    def init_dirs(self):
        # Create directories for checkpoints and summaries
        self.checkpoint_dir = os.path.join(self.config.experiment_dir, "checkpoints")
        self.summary_dir = os.path.join(self.config.experiment_dir, "summaries")

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

    def init_global_step(self):
        with tf.variable_scope('step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    def init_replay_memory(self):
        # Populate the replay memory with initial experience
        print("initializing replay memory...")

        state = self.environment.reset()
        for i in range(self.config.replay_memory_init_size):
            action_probs = self.policy(self.sess, state, self.epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = self.environment.step(self.sess, self.environment.valid_actions[action])
            self.memory.push(state, next_state, action, reward, done)
            if done:
                state = self.environment.reset()
            else:
                state = next_state

    def policy_fn(self, fn_type, estimator, n_actions):

        def epsilion_greedy(sess, observation, epsilon):
            actions = np.ones(n_actions, dtype=float) * epsilon / n_actions
            q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
            best_action = np.argmax(q_values)
            actions[best_action] += (1.0 - epsilon)
            return actions

        def greedy(sess, observation):
            q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
            best_action = np.argmax(q_values)
            return best_action

        if fn_type == 'epsilion_greedy':
            return epsilion_greedy
        elif fn_type == 'greedy':
            return greedy
        else:
            raise Exception("Please Select a proper policy function")

    def observe(self):
        pass

    def update_target_estimator(self, sess, q, target):
        # TODO copy paramters from q to target
        pass

    def train_episodic(self):

        self.epsilon = self.config.initial_epsilion
        self.epsilon_step = (self.config.initial_epsilion - self.config.final_epsilion) / self.config.exploration_steps
        self.policy = self.policy_fn(self.config.policy_fn, self.q_predictor, self.environment.n_actions)

        self.init_replay_memory()

        for cur_episode in range(self.config.num_episodes):

            # Save the current checkpoint
            self.save()

            # Reset the environment
            state = self.environment.reset()

            loss = 0

            # One step in the environment
            for t in itertools.count():

                # Epsilon for this time step
                self.epsilon = min(self.config.final_epsilion, self.epsilon - self.epsilon_step)

                self.episode_summary = tf.Summary()

                # Add epsilon to Tensorboard
                self.episode_summary.value.add(simple_value=self.epsilon, tag="epsilon")
                self.summary_writer.add_summary(self.episode_summary, self.global_step_tensor.eval())

                # Maybe update the target estimator
                if self.global_step_tensor.eval() % self.config.update_target_estimator_every == 0:
                    self.update_target_estimator(self.sess, self.q_predictor, self.target_estimator)
                    print("\nCopied model parameters to target network.")
###################### 
###################### new policy?   #omar
###################### 

                # Take a step
                action_probs = self.policy(self.sess, state, self.epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                next_state, reward, done = self.environment.step(action)

                # Save transition to replay memory
                self.memory.push(state, next_state, action, reward, done)

                # Update statistics

                # Sample a minibatch from the replay memory
                state_batch, next_state_batch, action_batch, reward_batch, done_batch = self.memory.get_batch(self.config.batch_size)

                # Calculate q values and targets
                q_values_next = self.target_estimator.predict(self.sess, next_state_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.config.discount_factor * np.amax(q_values_next, axis=1)

                # Perform gradient descent update
                loss = self.q_predictor.update(self.sess, state_batch, action_batch, targets_batch)

                if done:  # IF terminal state so exit the episode
                    break

                state = next_state

                self.global_step_assign_op.eval({self.global_step_input: self.global_step_tensor.eval() + 1})

                # Add summaries to tensorboard
                # TODO add summaries to tensorboard

        pass

    def train_continous(self):
        # TODO implement on global step only
        pass

    def play(self, n_episode=10):
        self.policy = self.policy_fn('greedy', self.q_predictor, self.environment.n_actions)

        for episode in range(n_episode):
            
            state = self.environment.reset()

            for t in itertools.count():

                best_action = self.policy(self.sess, state)
                next_state, reward, done = self.environment.step(best_action)

                if done:
                    break

                state = next_state
