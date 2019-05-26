from __future__ import absolute_import, division, print_function, unicode_literals
import ray
import ray.rllib.agents.a3c as a3c
from ray.rllib.models import ModelCatalog, Model
from ray.tune.registry import register_env

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from GomokuEnv import GomokuEnv
from ray import tune

BOARD_SIZE=10

class GomokuModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        shape = input_dict["obs"].shape
        self.model = Sequential()
        self.model.add(layers.InputLayer(
             input_tensor=tf.expand_dims(input_dict["obs"], axis=3),
             input_shape=(*shape, 1)))
        self.model.add(layers.Conv2D(8, (3, 3), name='l1', activation='relu'))
        self.model.add(layers.Conv2D(8, (3, 3), name='l2', activation='relu'))
        self.model.add(layers.Flatten(name='flat'))
        self.model.add(layers.Dense(int(shape[1])**2, name='last', activation='softmax'))
        return self.model.output, self.model.get_layer("flat").output


ray.init()
ModelCatalog.register_custom_model("GomokuModel", GomokuModel)
GomokuEnv_inst=GomokuEnv.GomokuEnv(BOARD_SIZE)
register_env("GomokuEnv", lambda _:GomokuEnv_inst)


trainer = a3c.A3CTrainer(env="GomokuEnv", config={
    "model": {"custom_model": "GomokuModel"},
},logger_creator=lambda _: ray.tune.logger.NoopLogger({},None))

print(trainer.train())
