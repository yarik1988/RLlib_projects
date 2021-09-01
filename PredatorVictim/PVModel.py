import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


class PredatorVictimModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(PredatorVictimModel, self).__init__(obs_space, action_space, num_outputs, model_config,name)
        input_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        hidden_layer1 = tf.keras.layers.Dense(500, activation='relu')(input_layer)
        output_mean = tf.keras.layers.Dense(2, activation='tanh')(hidden_layer1)
        output_std = tf.keras.layers.Dense(2, activation='sigmoid')(hidden_layer1)
        output_layer = tf.keras.layers.Concatenate(axis=1)([output_mean, output_std])
        value_layer = tf.keras.layers.Dense(1)(hidden_layer1)
        self.base_model = tf.keras.Model(input_layer, [output_layer, value_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])