from gym.envs.registration import register

register(
    id='OneDControl-v0',
    entry_point='OneDControl.control:ControlEnv',
)