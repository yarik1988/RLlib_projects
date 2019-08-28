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