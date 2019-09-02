import pickle
import os

def load_weights(trainer, BOARD_SIZE, NUM_IN_A_ROW):
    model_file = "weights_{}_{}.pickle".format(BOARD_SIZE, NUM_IN_A_ROW)
    if os.path.isfile(model_file):
        weights = pickle.load(open(model_file, "rb"))
        trainer.restore_from_object(weights)
        print("Model previous state loaded!")
    return trainer

def save_weights(trainer, BOARD_SIZE, NUM_IN_A_ROW):
    model_file = "weights_{}_{}.pickle".format(BOARD_SIZE, NUM_IN_A_ROW)
    weights = trainer.save_to_object()
    pickle.dump(weights, open(model_file, 'wb'))

def clb_episode_end(info):
    episode = info["episode"]
    episode.custom_metrics["agent_0_win_rate"] = episode.last_info_for("agent_0")["result"]
    episode.custom_metrics["agent_1_win_rate"] = episode.last_info_for("agent_1")["result"]
    episode.custom_metrics["game_duration"] = episode.last_info_for("agent_0")["nsteps"]+episode.last_info_for("agent_1")["nsteps"]
    episode.custom_metrics["wrong_moves"] = episode.last_info_for("agent_0")["wrong_moves"] \
                                            + episode.last_info_for("agent_1")["wrong_moves"]


def change_active_policy(trainer, policy):
    new_config = trainer.get_config()
    new_config['multiagent']["policies_to_train"] = [policy]
    trainer._setup(new_config)