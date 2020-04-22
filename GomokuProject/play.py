import pprint
import ray
from ray.rllib.models import ModelCatalog, Model
from ray.tune.registry import register_env
from GomokuEnv import GomokuEnv
import time
import aux_fn
import model_a3c as gm


PC_agents=['agent_0','agent_1']
PC_agents=['agent_0']
#PC_agents=[]
GENV=GomokuEnv.GomokuEnv(gm.BOARD_SIZE,gm.NUM_IN_A_ROW)
pp = pprint.PrettyPrinter(indent=4)

if len(PC_agents)>0:
    ray.init()
    ModelCatalog.register_custom_model("GomokuModel", gm.GomokuModel)
    register_env("GomokuEnv", lambda _:GENV)
    trainer = gm.get_trainer(GENV, 2)
    trainer = aux_fn.load_weights(trainer, gm.BOARD_SIZE, gm.NUM_IN_A_ROW)

obs = GENV.reset()
cur_action = None
rew = None
info = None
done = False
cur_action = {'agent_0':None,'agent_1':None}

while not done:
    cur_ag = 'agent_{}'.format(int(GENV.parity))
    policy_ag = (gm.map_fn(1))(cur_ag)
    obs_ag = obs[cur_ag]
    rew_ag = None if rew is None else rew['agent_{}'.format(int(GENV.parity))]
    if cur_ag in PC_agents:
       cur_action[cur_ag] = trainer.compute_action(obs_ag, prev_reward=rew_ag, prev_action=cur_action[cur_ag], policy_id=policy_ag)
    else:
       cur_action[cur_ag] = GENV.make_mouse_move()
    action_wrap = {cur_ag: cur_action[cur_ag]}
    obs, rew, dones, info = GENV.step(action_wrap)
    done=dones["__all__"]
    GENV.render()
    if len(PC_agents) == 2:
        time.sleep(0.5)

GENV.hmove=None
while GENV.hmove is None:
           GENV.render()
GENV.close()    
ray.shutdown()