import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

__version__ = 0.1


class BaseModel(object):
    """A Base Class for any model to be constructed"""

    def __init__(self):
        pass

    def build_model(self):
        raise NotImplemented()

    def predict(self, sess, state):
        raise NotImplemented()

    def update(self, sess, state, action, y):
        raise NotImplemented()


class Estimator(BaseModel):
    """Our Estimator Network"""

    def __init__(self):
        BaseModel.__init__(self)
        pass
