import VolleyEnv
from ray.rllib.models import ModelCatalog
import pygame
import gym
params = {'predator': {'max_vel': 0.01, 'max_acceleration': 0.001},
          'victim': {'max_vel': 0.002, 'max_acceleration': 0.0001},
          'reward_scale': 0.01,
          'max_steps': 1000,
          'is_continuous': False,
          'catch_distance': 0.1}
PVEnv = gym.make("VolleyEnv-v0", params=params)
PVEnv.reset()
pygame.init()
is_running=True
while is_running:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        PVEnv.move_left(1)
    if keys[pygame.K_RIGHT]:
        PVEnv.move_right(1)
    if keys[pygame.K_UP]:
        PVEnv.jump(1)

    if keys[pygame.K_a]:
        PVEnv.move_left(0)
    if keys[pygame.K_d]:
        PVEnv.move_right(0)
    if keys[pygame.K_w]:
        PVEnv.jump(0)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            is_running = False
    PVEnv.step(dict())
    is_running = is_running and PVEnv.render()

PVEnv.close()

