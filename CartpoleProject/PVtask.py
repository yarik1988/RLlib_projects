import numpy as np
from PredatorVictim import PredatorVictim
params={'max_predator_vel': 0.01, 'max_victim_vel': 0.01, 'max_steps': 2000000}
PVEnv = PredatorVictim.PredatorVictim(params)
PVEnv.reset()
done = False
while not done:
    obs, rewards, dones, info = PVEnv.step({"predator": (np.random.rand(2)*2-1)*0.001, "victim": (np.random.rand(2)*2-1)*0.001})
    done = dones['__all__']
    PVEnv.render()
    print(PVEnv.n_steps)

