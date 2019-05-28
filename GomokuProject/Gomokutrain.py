from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import ray
import ray.rllib.agents.a3c as a3c
from ray.rllib.models import ModelCatalog, Model
from ray.tune.registry import register_env

from tensorflow.keras import layers, Sequential
from GomokuEnv import GomokuEnv
import pickle
import time
import _thread
import psutil
from datetime import timedelta

def input_thread(a_list):
    input()
    a_list.append(True)
BOARD_SIZE=10
NUM_IN_A_ROW=5
model_file = "weights_{}_{}.pickle".format(BOARD_SIZE,NUM_IN_A_ROW)

class GomokuModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        shape = input_dict["obs"]["real_obs"].shape
        self.model = Sequential()
        self.model.add(layers.InputLayer(
             input_tensor=tf.expand_dims(input_dict["obs"]["real_obs"], axis=3),
             input_shape=(*shape, 1)))
        self.model.add(layers.Conv2D(16, (3, 3), name='l1', activation='relu'))
        self.model.add(layers.Conv2D(32, (3, 3), name='l2', activation='relu'))
        self.model.add(layers.Conv2D(32, (3, 3), name='l3', activation='relu'))
        self.model.add(layers.Conv2D(32, (3, 3), name='l4', activation='relu'))
        self.model.add(layers.Flatten(name='flat'))
        self.model.add(layers.Dense(int(shape[1])**2, name='last', activation='softmax'))
        self.model.summary()
        inf_mask = tf.maximum(tf.math.log(input_dict["obs"]["action_mask"]), tf.float32.min)
        output = self.model.output+inf_mask
        return output, self.model.get_layer("flat").output


ray.init()
ModelCatalog.register_custom_model("GomokuModel", GomokuModel)
GENV=GomokuEnv.GomokuEnv(BOARD_SIZE,NUM_IN_A_ROW)
register_env("GomokuEnv", lambda _:GENV)


def gen_policy(i):
    config = {
        "model": {
            "custom_model": 'GomokuModel',
        }
    }
    return (None, GENV.observation_space, GENV.action_space, config)


# Setup PPO with an ensemble of `num_policies` different policies
policies = {
    "policy_{}".format(i): gen_policy(i)
    for i in range(2)
}
policy_ids = list(policies.keys())


def map_fn(agent_id):
    if agent_id=='agent_1':
        return policy_ids[0]
    else:
        return policy_ids[1]


trainer = a3c.A3CTrainer(env="GomokuEnv", config={
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": map_fn

    },
}, logger_creator=lambda _: ray.tune.logger.NoopLogger({},None))

if os.path.isfile(model_file):
   weights = pickle.load(open(model_file, "rb"))
   trainer.restore_from_object(weights)
   print("Model previous state loaded!")


a_list = []
_thread.start_new_thread(input_thread, (a_list,))
start = time. time()
while not a_list:
    rest = trainer.train()
    mem_info = psutil.virtual_memory()
    if mem_info.percent > 90:
        a_list.append(True)
    print("Memory usage = {}".format(mem_info.percent))
    print("Episode reward mean = {}".format(rest["episode_reward_mean"]))
    print(rest["info"]["learner"])

weights = trainer.save_to_object()
pickle.dump(weights, open(model_file, 'wb'))
print("Execution time: {}".format(timedelta(seconds=time.time()-start)))

obs = GENV.reset()
cur_action = None
rew = None
info = None
done = False
parity = False
cur_action = {'agent_0':None,'agent_1':None}

while not done:
    cur_ag='agent_{}'.format(int(parity))
    policy_ag = 'policy_{}'.format(int(parity))
    obs_ag = obs[cur_ag]
    rew_ag = None if rew is None else rew['agent_{}'.format(int(parity))]
    cur_action[cur_ag] = trainer.compute_action(obs_ag, prev_reward=rew_ag, prev_action=cur_action[cur_ag], policy_id=policy_ag)
    action_wrap = {cur_ag: cur_action[cur_ag]}
    obs, rew, dones, info = GENV.step(action_wrap)
    done=dones["__all__"]
    GENV.render()
    parity = not parity
    time.sleep(1)

GENV.close()
ray.shutdown()
