"""This file is to put any configuration to our classes"""

class Configurations(object):
	"""docstring for configuration"""
	def __init__(self):
		self.experiment_dir = "./expriment"

		#Environment config
		self.env_name = 'Breakout-v0'
		self.state_processor_params = { "resize_shape": (84, 84),
										"crop_box": (34, 0, 160, 160),
										"gray": True,
										"frames_num": 4 }
		self.record_video_every = 10


		#Agent config



		#Replay_Memory config



		#Estimator config		