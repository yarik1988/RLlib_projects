import ray
import sys, select
from ray.tune.registry import register_env
from GomokuEnv import GomokuEnv
import time
import pprint
import psutil
import aux_fn
import model_dqn as gm
num_policies = 2

ray.init()
GENV = GomokuEnv.GomokuEnv(gm.BOARD_SIZE, gm.NUM_IN_A_ROW)
register_env("GomokuEnv", lambda _: GENV)
trainer = gm.get_trainer(GENV, num_policies)

trainer = aux_fn.load_weights(trainer, gm.BOARD_SIZE, gm.NUM_IN_A_ROW)
start = time.time()
pp = pprint.PrettyPrinter(indent=4)

while True:
    rest = trainer.train()
    mem_info = psutil.virtual_memory()
    print("Memory usage = {}".format(mem_info.percent))
    if 'learner' in rest['info']:
        if 'policy_0' in rest['info']['learner']:
            print("First policy learning rate={}".format(rest['info']['learner']['policy_0']['cur_lr']))
        if 'policy_1' in rest['info']['learner']:
            print("Second policy learning rate={}".format(rest['info']['learner']['policy_1']['cur_lr']))
    if 'agent_0_win_rate_mean' in rest['custom_metrics']:
        print("First player win rate = {}%".format(100*rest['custom_metrics']['agent_0_win_rate_mean']))
    if 'agent_1_win_rate_mean' in rest['custom_metrics']:
        print("Second player win rate = {}%".format(100*rest['custom_metrics']['agent_1_win_rate_mean']))
    if 'game_duration_min' in rest['custom_metrics']:
        print("Minimum game duration = {}".format(rest['custom_metrics']['game_duration_min']))
    if 'game_duration_mean' in rest['custom_metrics']:
        print("Average game duration = {}".format(rest['custom_metrics']['game_duration_mean']))
    if 'game_duration_max' in rest['custom_metrics']:
        print("Maximum game duration = {}".format(rest['custom_metrics']['game_duration_max']))
    if 'wrong_moves_mean' in rest['custom_metrics']:
        print("Wrong moves count = {}".format(rest['custom_metrics']['wrong_moves_mean']))
    if time.time()-start > 300:
        print('Weights saving')
        aux_fn.save_weights(trainer, gm.BOARD_SIZE, gm.NUM_IN_A_ROW)
        start = time.time()
    if mem_info.percent > 90:
        print("Restarting trainer")
        state = trainer.save(".")
        trainer.stop()
        trainer = gm.get_trainer(GENV, num_policies)
        trainer.restore(state)

    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        break

aux_fn.save_weights(trainer, gm.BOARD_SIZE, gm.NUM_IN_A_ROW)
ray.shutdown()
