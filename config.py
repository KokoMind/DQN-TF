"""This file is to put any configuration to our classes"""


class EnvConfig(object):
    env_name = 'Breakout-v0'
    state_processor_params = {"resize_shape": (84, 84),
                              "crop_box": (34, 0, 160, 160),
                              "gray": True,
                              "frames_num": 4}
    monitor = True
    record_video_every = 10


class AgentConfig(object):
    pass


class ReplayMemoryConfig(object):
    rep_max_size = 500000


class EstimatorConfig(object):
    name = "DQN_Dragon"
    learning_rate = .0002


class Experiment1(EnvConfig, AgentConfig, ReplayMemoryConfig, EstimatorConfig):
    is_train = None
    initial_training = None
    cont_training = None
    is_play = None
    state_shape = [84, 84, 4]

    experiment_dir = "./expriment_1/"


def get_config():
    return Experiment1
