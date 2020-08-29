import cv2


def evaluate(trainer, PVEnv, n_iter=5, video_file=None):
    if video_file is not None:
        video = cv2.VideoWriter("../videos/Predator_Victim.avi", 0, 60, (PVEnv.screen_wh, PVEnv.screen_wh))
    for i in range(n_iter):
        obs = PVEnv.reset()
        done = False
        while not done:
            action_predator = trainer.compute_action(obs['predator'], policy_id="policy_predator", explore=False)
            action_victim = trainer.compute_action(obs['victim'], policy_id="policy_predator", explore=False)
            obs, rewards, dones, info = PVEnv.step({"predator": action_predator, "victim": action_victim})
            done = dones['__all__']
            frame = PVEnv.render(mode='rgb_array')
            if video_file is not None:
                video.write(frame[..., ::-1])
        PVEnv.close()
    if video_file is not None:
        video.release()
