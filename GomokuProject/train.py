import ray
import sys,select
from ray.tune.registry import register_env
from GomokuEnv import GomokuEnv
import time
import pprint
import psutil
import aux_fn
import model_a3c as gm
num_policies=2

ray.init()
GENV = GomokuEnv.GomokuEnv(gm.BOARD_SIZE, gm.NUM_IN_A_ROW)
register_env("GomokuEnv", lambda _: GENV)
trainer = gm.get_trainer(GENV,num_policies)
trainer = aux_fn.load_weights(trainer,gm.BOARD_SIZE,gm.NUM_IN_A_ROW)

start = time.time()
pp = pprint.PrettyPrinter(indent=4)
if num_policies>1:
    cur_switch=False
    aux_fn.change_active_policy(trainer,"policy_{}".format(int(cur_switch)))
    subepoch_count = 0

while True:
    rest = trainer.train()
    mem_info = psutil.virtual_memory()
    print("Memory usage = {}".format(mem_info.percent))
    if 'policy_0' in rest['info']['learner']:
        print("First policy learning rate={}".format(rest['info']['learner']['policy_0']['cur_lr']))
    if 'policy_1' in rest['info']['learner']:
        print("Second policy learning rate={}".format(rest['info']['learner']['policy_1']['cur_lr']))
    print("First player win rate = {}%".format(100*rest['custom_metrics']['agent_0_win_rate_mean']))
    print("Second player win rate = {}%".format(100*rest['custom_metrics']['agent_1_win_rate_mean']))
    print("Minimum game duration = {}".format(rest['custom_metrics']['game_duration_min']))
    print("Average game duration = {}".format(rest['custom_metrics']['game_duration_mean']))
    print("Maximum game duration = {}".format(rest['custom_metrics']['game_duration_max']))
    print("Wrong moves count = {}".format(rest['custom_metrics']['wrong_moves_mean']))
    if time.time()-start>300:
          print('Weights saving')
          aux_fn.save_weights(trainer, gm.BOARD_SIZE, gm.NUM_IN_A_ROW)
          start = time.time()
    if mem_info.percent > 90 or sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        break
    if num_policies >1:
        subepoch_count += 1
        print("Subepoch count = {}".format(subepoch_count))
        ohter_ag = 'agent_{}_win_rate_mean'.format((int(not cur_switch)))
        other_ag_win = rest['custom_metrics'][ohter_ag]
        if (other_ag_win < 0.05 and subepoch_count > 50) or subepoch_count > 400:
            print("Change learning policy. Restarting trainer")
            cur_switch = not cur_switch
            state = trainer.save()
            trainer.stop()
            trainer = gm.get_trainer(GENV, num_policies,["policy_{}".format(int(cur_switch))])
            trainer.restore(state)
            subepoch_count = 0

aux_fn.save_weights(trainer, gm.BOARD_SIZE, gm.NUM_IN_A_ROW)
ray.shutdown()
