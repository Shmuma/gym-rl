import gym
import numpy as np
import random


class ReplayGenerator:
    def __init__(self, env, dqn, gamma, batch_size):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.current_state = self.env.reset()
        self.dqn = dqn
        # statistics of generator
        self.episodes = 0


    def next(self):
        """
        Generate next batch
        :return: tuple of (states, action, reward, next_states) arrays
        """
        states = []
        actions = []
        rewards = []
        next_states = []

        while len(states) < self.batch_size:
            # choose action
            if random.random() < self.gamma:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.dqn.calc_qvals(self.current_state))

            observation, reward, done, _ = self.env.step(action)
            states.append(self.current_state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(observation)

            if done:
                self.episodes += 1
                self.current_state = self.env.reset()
            else:
                self.current_state = observation

        return states, actions, rewards, next_states
