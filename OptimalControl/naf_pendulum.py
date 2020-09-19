import numpy as np
import os.path
import scipy.io
import gym
from gym import wrappers
import SimpleControl
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import *
import rllib
from rllib.agents import NAFAgent
from rllib.memory import SequentialMemory
from rllib.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.callbacks import *

ENV_NAME = 'SimpleControl-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
nb_actions = env.action_space.shape[0]
act_fun = 'relu'
act_finlayer = 'linear'
# Build all necessary models: V, mu, and L networks.
V_model = Sequential()
V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
V_model.add(Dense(16))
V_model.add(Activation(act_fun))
V_model.add(Dense(16))
V_model.add(Activation(act_fun))
V_model.add(Dense(16))
V_model.add(Activation(act_fun))
V_model.add(Dense(1))
V_model.add(Activation(act_finlayer))
print(V_model.summary())

mu_model = Sequential()
mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
mu_model.add(Dense(16))
mu_model.add(Activation(act_fun))
mu_model.add(Dense(16))
mu_model.add(Activation(act_fun))
mu_model.add(Dense(16))
mu_model.add(Activation(act_fun))
mu_model.add(Dense(nb_actions))
mu_model.add(Activation(act_finlayer))
print(mu_model.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
x = Concatenate()([action_input, Flatten()(observation_input)])
x = Dense(32)(x)
x = Activation(act_fun)(x)
x = Dense(32)(x)
x = Activation(act_fun)(x)
x = Dense(32)(x)
x = Activation(act_fun)(x)
x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
x = Activation(act_finlayer)(x)
L_model = Model(inputs=[action_input, observation_input], outputs=x)
print(L_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=0.15, mu=0., sigma=0.3, size=nb_actions)
agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=100, random_process=random_process,
                 gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
fname='cdqn_{}_weights.h5f'.format(ENV_NAME)
if os.path.isfile(fname):
    agent.load_weights(fname)

#agent.fit(env, nb_steps=100000, visualize=False, verbose=1000, nb_max_episode_steps=200)

# After training is done, we save the final weights.
#agent.save_weights(fname, overwrite=True)

class MyLogger(Callback):

    def __init__(self):
        self.pt_position = list()

    def on_step_end(self, step, logs={}):
        self.pt_position.append(self.model.recent_observation)

    def on_episode_end(self, step, logs={}):
        scipy.io.savemat('trajectory.mat', mdict={'trajectory': self.pt_position})



# Finally, evaluate our algorithm for 5 episodes.
for i in range(1,5):
    env = wrappers.Monitor(env, 'video/', resume=True)
    agent.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=500,callbacks=[MyLogger()])
    wrappers.Monitor.close(env)

