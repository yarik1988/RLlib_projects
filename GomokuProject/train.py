import os
import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog, Model
from ray.tune.registry import register_env
from GomokuEnv import GomokuEnv
import pickle
import time
import _thread
import psutil
import gomoku_model


def input_thread(a_list):
    input()
    a_list.append(True)

BOARD_SIZE=3
NUM_IN_A_ROW=3

ray.init()
ModelCatalog.register_custom_model("GomokuModel",gomoku_model.GomokuModel)

GENV=GomokuEnv.GomokuEnv(BOARD_SIZE,NUM_IN_A_ROW)
register_env("GomokuEnv", lambda _:GENV)



trainer = ppo.APPOTrainer(env="GomokuEnv", config={
    "multiagent": {
        "policies": {"policy_{}".format(i): gomoku_model.gen_policy(GENV) for i in range(2)},
        "policy_mapping_fn": gomoku_model.map_fn

    },
}, logger_creator=lambda _: ray.tune.logger.NoopLogger({},None))


model_file = "weights_{}_{}.pickle".format(BOARD_SIZE,NUM_IN_A_ROW)

if os.path.isfile(model_file):
   weights = pickle.load(open(model_file, "rb"))
   trainer.restore_from_object(weights)
   print("Model previous state loaded!")


a_list = []
_thread.start_new_thread(input_thread, (a_list,))
start = time.time()
while not a_list:
    rest = trainer.train()
    mem_info = psutil.virtual_memory()
    if mem_info.percent > 90:
        a_list.append(True)
    print("Memory usage = {}".format(mem_info.percent))
    print("Episode reward mean = {}".format(rest["episode_reward_mean"]))
    print(rest["info"]["learner"])
    if time.time()-start>300:
          print('Weights saving')
          weights = trainer.save_to_object()
          pickle.dump(weights, open(model_file, 'wb'))
          start = time.time()

weights = trainer.save_to_object()
pickle.dump(weights, open(model_file, 'wb'))

GENV.close()
ray.shutdown()
