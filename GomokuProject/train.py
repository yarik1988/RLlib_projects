import ray
from ray.tune.registry import register_env
from GomokuEnv import GomokuEnv
import time
import pprint
import _thread
import psutil
import gomoku_model

def input_thread(a_list):
    input()
    a_list.append(True)

BOARD_SIZE = 3
NUM_IN_A_ROW = 3

ray.init()
GENV = GomokuEnv.GomokuEnv(BOARD_SIZE, NUM_IN_A_ROW)
register_env("GomokuEnv", lambda _: GENV)
trainer = gomoku_model.get_trainer(GENV)
trainer=gomoku_model.load_weights(trainer,BOARD_SIZE,NUM_IN_A_ROW)

a_list = []
_thread.start_new_thread(input_thread, (a_list,))
start = time.time()
pp = pprint.PrettyPrinter(indent=4)
while not a_list:
    rest = trainer.train()
    mem_info = psutil.virtual_memory()
    if mem_info.percent > 90:
        a_list.append(True)
    print("Memory usage = {}".format(mem_info.percent))
    print("Episode reward mean = {}".format(rest["episode_reward_mean"]))
    pp.pprint(rest["info"]["learner"])
    if time.time()-start>300:
          print('Weights saving')
          gomoku_model.save_weights(trainer, BOARD_SIZE, NUM_IN_A_ROW)
          start = time.time()

gomoku_model.save_weights(trainer, BOARD_SIZE, NUM_IN_A_ROW)
ray.shutdown()
