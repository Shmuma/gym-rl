import gym
import tensorflow as tf
import argparse

from lib import dqn, replay

BATCH_SIZE = 16

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    net = dqn.DenseDQN(input_size, output_size, batch_size=BATCH_SIZE, neurons=(128, 64))
    replay_generator = replay.ReplayGenerator(env, net, gamma=0.0, batch_size=BATCH_SIZE)

    print net

    with tf.Session() as session:
        session.run([tf.initialize_all_variables()])
        batch = replay_generator.next()
        qvals = net.calc_qvals(batch[0][1])
        print qvals