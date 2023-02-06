from abc import ABC

from ray.rllib.models.tf import TFModelV2

import tensorflow as tf
class CartpoleModel(TFModelV2, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CartpoleModel, self).__init__(obs_space, action_space, num_outputs, model_config,name)
        input_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        hidden_layer=input_layer
        for i in range(3):
            hidden_layer = tf.keras.layers.Dense(256, activation='tanh',
                                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01,l2=0.0001),
                                                 bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01,l2=0.0001))(hidden_layer)
        output_layer = tf.keras.layers.Dense(num_outputs)(hidden_layer)
        value_layer = tf.keras.layers.Dense(1)(hidden_layer)
        self.base_model = tf.keras.Model(input_layer, [output_layer, value_layer])

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])