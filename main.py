""" Let's Begin the action :P  """

import tensorflow as tf

from tensorflow import reset_default_graph
from environment import Environment
from agent import Agent
from config import get_config


def main():
    # Reset the graph
    reset_default_graph()

    # Get the Config of the program
    config = get_config()

    # Create the Session of the graph
    sess = tf.Session()

    env = Environment(sess, config)

    wasted = Agent(sess, config, env)

    if config.is_train:
        try:
            wasted.train_episodic()
        except KeyboardInterrupt:
            wasted.save()
    elif config.is_play:
        wasted.play()
    else:
        raise Exception("Please select a proper mode for our wasted agent")

    sess.close()


if __name__ == '__main__':
    main()
