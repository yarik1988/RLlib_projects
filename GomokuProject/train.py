import ray
import sys,select
from ray.tune.registry import register_env
from GomokuEnv import GomokuEnv
import time
import pprint
import psutil
import gomoku_model

BOARD_SIZE = 3
NUM_IN_A_ROW = 3

ray.init()
GENV = GomokuEnv.GomokuEnv(BOARD_SIZE, NUM_IN_A_ROW)
register_env("GomokuEnv", lambda _: GENV)
trainer = gomoku_model.get_trainer(GENV)
trainer=gomoku_model.load_weights(trainer,BOARD_SIZE,NUM_IN_A_ROW)

start = time.time()
pp = pprint.PrettyPrinter(indent=4)
while True:
    rest = trainer.train()
    mem_info = psutil.virtual_memory()
    print("Memory usage = {}".format(mem_info.percent))
    print("First player win rate = {}%".format(100*rest['custom_metrics']['agent_0_win_rate_mean']))
    print("Second player win rate = {}%".format(100*rest['custom_metrics']['agent_1_win_rate_mean']))
    if time.time()-start>300:
          print('Weights saving')
          gomoku_model.save_weights(trainer, BOARD_SIZE, NUM_IN_A_ROW)
          start = time.time()
    if mem_info.percent > 90 or sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        break


gomoku_model.save_weights(trainer, BOARD_SIZE, NUM_IN_A_ROW)
ray.shutdown()
