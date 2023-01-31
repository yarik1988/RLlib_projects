import os
import pickle
import keyboard
import numpy as np
import gym
import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ddpg as ddpg
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.tune.registry import register_env
import tensorflow as tf
from PredatorVictim import PredatorVictim
from evaluate import evaluate


ready_to_exit = False
def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit=True


class PredatorVictimModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(PredatorVictimModel, self).__init__(obs_space, action_space, num_outputs, model_config,name)
        input_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        hidden_layer1 = tf.keras.layers.Dense(500, activation='relu')(input_layer)
        output_mean = tf.keras.layers.Dense(2, activation='tanh')(hidden_layer1)
        output_std = tf.keras.layers.Dense(2, activation='sigmoid')(hidden_layer1)
        output_layer = tf.keras.layers.Concatenate(axis=1)([output_mean, output_std])
        value_layer = tf.keras.layers.Dense(1)(hidden_layer1)
        self.base_model = tf.keras.Model(input_layer, [output_layer, value_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def gen_policy(PVEnv, i):
    ModelCatalog.register_custom_model("PredatorVictimModel_{}".format(i), PredatorVictimModel)
    config = {
        "model": {"custom_model": "PredatorVictimModel_{}".format(i)},
    }
    return None, PVEnv.observation_space, PVEnv.action_space, config


def policy_mapping_fn(agent_id):
    if agent_id == 'predator':
        return "policy_predator"
    elif agent_id == 'victim':
        return "policy_victim"


model_file = 'PredatorVictim_CA.pickle'
params = {'predator': {'max_vel': 0.01, 'max_acceleration': 0.001},
          'victim': {'max_vel': 0.002, 'max_acceleration': 0.0001},
          'reward_scale': 0.01,
          'max_steps': 1000,
          'is_continuous': True,
          'catch_distance': 0.1}


ray.init(include_dashboard=False)
ModelCatalog.register_custom_model("CartpoleModel", PredatorVictimModel)
PVEnv = gym.make("PredatorVictim-v0", params=params)
register_env("PredatorVictimEnv", lambda _: PVEnv)

trainer = a3c.A3CTrainer(env="PredatorVictimEnv", config={
        "multiagent": {
            "policies": {"policy_predator": gen_policy(PVEnv, 0),
                         "policy_victim": gen_policy(PVEnv, 1)},
            "policy_mapping_fn": policy_mapping_fn,
            },
    })

if os.path.isfile(model_file):
    weights = pickle.load(open(model_file, "rb"))
    trainer.restore_from_object(weights)


keyboard.on_press_key("q", press_key_exit)
while True:
    if ready_to_exit:
        break
    rest = trainer.train()
    print(rest['policy_reward_mean'])

weights = trainer.save_to_object()
pickle.dump(weights, open(model_file, 'wb'))
print('Model saved')


evaluate(trainer, PVEnv, video_file='../videos/Predator_Victim_A3C')