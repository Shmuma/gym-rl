import gym
import tensorflow as tf
import argparse

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    for episode in range(20):
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