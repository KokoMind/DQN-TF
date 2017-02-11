"""This file is to put any configuration to our classes"""


class EnvConfig():
    env_name = 'Breakout-v0'
    state_processor_params = {"resize_shape": (84, 84),
                              "crop_box": (34, 0, 160, 160),
                              "gray": True,
                              "frames_num": 4}
    monitor = True
    record_video_every = 10


class AgentConfig():
    bla = 'bla'


class ReplayMemoryConfig():
    def __init__(self):
        self.max_size = 500000


class EstimatorConfig():
    def __init__(self):
        self.name = "blablabla"
        self.learning_rate = .0002


class Expriment1(EnvConfig, AgentConfig, ReplayMemoryConfig, EstimatorConfig):
    experiment_dir = "./expriment/"


def get_config():
    return Expriment1
