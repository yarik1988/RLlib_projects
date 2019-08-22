import os
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
from ray.rllib.models import ModelCatalog, Model
from ray.tune.registry import register_env
from GomokuEnv import GomokuEnv
import pickle
import gomoku_model
import time

BOARD_SIZE=3
NUM_IN_A_ROW=3

PC_agents=['agent_0','agent_1']
#PC_agents=['agent_0']
GENV=GomokuEnv.GomokuEnv(BOARD_SIZE,NUM_IN_A_ROW)


if len(PC_agents)>0:
    ray.init()
    ModelCatalog.register_custom_model("GomokuModel",gomoku_model.GomokuModel)
    register_env("GomokuEnv", lambda _:GENV)

    trainer = a3c.A3CTrainer(env="GomokuEnv", config={
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

obs = GENV.reset()
cur_action = None
rew = None
info = None
done = False
cur_action = {'agent_0':None,'agent_1':None}

while not done:
    cur_ag='agent_{}'.format(int(GENV.parity))
    policy_ag = 'policy_{}'.format(int(GENV.parity))
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
    if len(PC_agents)==2:
        time.sleep(0.5)

GENV.hmove=None
while GENV.hmove is None:
           GENV.render()
GENV.close()    
ray.shutdown()