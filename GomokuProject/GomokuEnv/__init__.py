from gym.envs.registration import register

register(
    id='GomokuEnv-v0',
    entry_point='GomokuEnv.GomokuEnv:GomokuEnv',
)