import gym
import tensorflow as tf
import argparse

from lib import dqn

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    net = dqn.DenseDQN(input_size, output_size, batch_size=16, neurons=(128, 64))

    print net

    for episode in range(1):
        observation = env.reset()
        step = 0
        while True:
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            print reward
            if done:
                print("Episode {} done in {} steps".format(episode+1, step+1))
                break
            step += 1