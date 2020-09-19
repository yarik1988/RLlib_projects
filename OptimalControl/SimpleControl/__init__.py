from gym.envs.registration import register

register(
    id='SimpleControl-v0',
    entry_point='SimpleControl.control:ControlEnv',
)