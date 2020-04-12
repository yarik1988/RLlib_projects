import ray
import time
import tensorflow as tf
import ray.rllib.agents.a3c as a3c
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.tune.registry import register_env
import gym
from gym import RewardWrapper
import pickle
import os
import _thread

def input_thread(a_list):
    input()
    a_list.append(True)


class ScaleReward(RewardWrapper):
    def reward(self, reward):
        return reward/100


class CartpoleModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CartpoleModel, self).__init__(obs_space, action_space, num_outputs, model_config,name)
        input_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        hidden_layer1 = tf.keras.layers.Dense(15, activation='tanh')(input_layer)
        hidden_layer2 = tf.keras.layers.Dense(15, activation='tanh')(hidden_layer1)
        hidden_layer3 = tf.keras.layers.Dense(30, activation='tanh')(hidden_layer2)
        output_layer = tf.keras.layers.Dense(num_outputs)(hidden_layer3)
        value_layer = tf.keras.layers.Dense(1)(hidden_layer3)
        self.base_model = tf.keras.Model(input_layer, [output_layer, value_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


ray.init()
ModelCatalog.register_custom_model("CartpoleModel", CartpoleModel)
CartpoleEnv = gym.make('CartPole-v0')
CartpoleEnv = ScaleReward(CartpoleEnv)
register_env("CP", lambda _:CartpoleEnv)



trainer = a3c.A3CTrainer(env="CP", config={"model": {"custom_model": "CartpoleModel"}})

if os.path.isfile('weights.pickle'):
   weights = pickle.load(open("weights.pickle", "rb"))
   trainer.restore_from_object(weights)


a_list = []
_thread.start_new_thread(input_thread, (a_list,))
while not a_list:
      rest=trainer.train()
      print(rest["episode_reward_mean"])

weights=trainer.save_to_object()
pickle.dump(weights, open('weights.pickle', 'wb'))
print('Model saved')


obs=CartpoleEnv.reset()
cur_action = None
total_rev = 0
rew = None
info = None
done = False
while not done:
    cur_action = trainer.compute_action(obs,prev_action=cur_action,prev_reward=rew,info=info)
    obs, rew, done, info = CartpoleEnv.step(cur_action)
    total_rev += rew
    CartpoleEnv.render()
    time.sleep(0.05)
print(total_rev)
CartpoleEnv.close()
ray.shutdown()
