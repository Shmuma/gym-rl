import gym

env = gym.make('CartPole-v0')
env.reset()

for t in range(100):
#    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print action, obs, reward, done
    if done:
        print ("Simulation done after {} steps".format(t+1))
        break