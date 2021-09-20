from gym.envs.registration import register

register(
    id='VolleyEnv-v0',
    entry_point='VolleyEnv.VolleyEnv:VolleyEnv',
)