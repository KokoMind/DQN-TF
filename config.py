"""This file is to put any configuration to our classes"""


class EnvConfig(object):
    env_name = 'Breakout-v0'
    state_shape = [84, 84, 4]
    state_processor_params = {"resize_shape": (84, 84),
                              "crop_box": (34, 0, 160, 160),
                              "gray": True,
                              "frames_num": 4}
    record_video_every = 5


class AgentConfig(object):
    initial_epsilon = 1.0
    final_epsilon = 0.1
    exploration_steps = 500000
    policy_fn = 'epsilon_greedy'
    discount_factor = 0.99
    evaluate_every = 25
    evaluation_episodes = 5


class ReplayMemoryConfig(object):
    rep_max_size = 100000
    replay_memory_init_size = 10000


class EstimatorConfig(object):
    name = "DQN_Dragon"
    learning_rate = 0.00025


class Experiment1(EnvConfig, AgentConfig, ReplayMemoryConfig, EstimatorConfig):
    is_train = True
    cont_training = True
    is_play = False
    num_episodes = 10000
    update_target_estimator_every = 10000
    batch_size = 32

    experiment_dir = "./experiment_1/"


def get_config():
    return Experiment1
