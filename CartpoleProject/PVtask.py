import os
import pickle
import keyboard
import numpy as np
import ray
import ray.rllib.agents.a3c as a3c
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.tune.registry import register_env
import tensorflow as tf
from PredatorVictim import PredatorVictim


ready_to_exit = False
def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit=True


class PredatorVictimModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(PredatorVictimModel, self).__init__(obs_space, action_space, num_outputs, model_config,name)
        input_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        hidden_layer1 = tf.keras.layers.Dense(100, activation='tanh')(input_layer)
        hidden_layer2 = tf.keras.layers.Dense(10, activation='tanh')(hidden_layer1)
        output_layer = tf.keras.layers.Dense(num_outputs, activation='tanh')(hidden_layer2)
        value_layer = tf.keras.layers.Dense(1)(hidden_layer2)
        self.base_model = tf.keras.Model(input_layer, [output_layer, value_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def policy_variables(self):
        """Return the list of variables for the policy net."""
        return list(self.action_net.variables)


def gen_policy(PVEnv, i):
    ModelCatalog.register_custom_model("PredatorVictimModel_{}".format(i), PredatorVictimModel)
    config = {
        "model": {"custom_model": "PredatorVictimModel_{}".format(i)},
    }
    return None, PVEnv.observation_space, PVEnv.action_space, config


params = {'max_predator_vel': 0.01,
          'max_victim_vel': 0.01,
          'max_predator_acceleration': 0.001,
          'max_victim_acceleration': 0.002,
          'max_steps': 1000,
          'catch_distance': 0.1}

ray.init(include_dashboard=False)
ModelCatalog.register_custom_model("CartpoleModel", PredatorVictimModel)
PVEnv = PredatorVictim.PredatorVictim(params)
register_env("PredatorVictimEnv", lambda _: PVEnv)

trainer = ray.rllib.agents.a3c.A3CTrainer(env="PredatorVictimEnv", config={
        "multiagent": {
            "policies": {"policy_predator": gen_policy(PVEnv, 0),
                         "policy_victim": gen_policy(PVEnv, 1)},
            "policy_mapping_fn": lambda agent_id: "policy_predator" if agent_id == 'predator' else "policy_victim",
            },
    })

if os.path.isfile('PredatorVictim.pickle'):
    weights = pickle.load(open("PredatorVictim.pickle", "rb"))
    trainer.restore_from_object(weights)

keyboard.on_press_key("q", press_key_exit)
while True:
    if ready_to_exit:
        break
    rest = trainer.train()
    print(rest['policy_reward_mean'])

weights = trainer.save_to_object()
pickle.dump(weights, open('PredatorVictim.pickle', 'wb'))
print('Model saved')


obs = PVEnv.reset()
cur_action = None
done = False
rew = None
info = None
done = False
while not done:
    action_predator = trainer.compute_action(obs['predator'], policy_id="policy_predator")
    action_victim = trainer.compute_action(obs['predator'], policy_id="policy_predator")
    obs, rewards, dones, info = PVEnv.step({"predator":action_predator,"victim":action_victim})
    done = dones['__all__']
    PVEnv.render()

