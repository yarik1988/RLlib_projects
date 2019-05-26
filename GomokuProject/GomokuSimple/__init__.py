from gym.envs.registration import register

register(
    id='GomokuSimple-v0',
    entry_point='GomokuSimple.GomokuSimple:GomokuSimple',
)