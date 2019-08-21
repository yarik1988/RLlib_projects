from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils import try_import_tf

class GomokuModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,name):
        super(GomokuModel, self).__init__(obs_space, action_space,num_outputs, model_config, name)
        with tf.compat.v1.variable_scope(
                tf.compat.v1.VariableScope(tf.compat.v1.AUTO_REUSE, "shared"),
                reuse=tf.compat.v1.AUTO_REUSE,
                auxiliary_name_scope=False):
            input_shp = obs_space.original_space.spaces['real_obs']
            self.inputs = tf.keras.layers.Input(shape=input_shp.shape, name="observations")
            layer_0 = tf.keras.layers.Flatten(name='flatlayer')(self.inputs)
            layer_1 = tf.keras.layers.Dense(50, name="my_layer1",activation=tf.nn.relu,kernel_initializer=normc_initializer(1.0))(layer_0)
            layer_2 = tf.keras.layers.Dense(50, name="my_layer2", activation=tf.nn.relu,kernel_initializer=normc_initializer(1.0))(layer_1)
            layer_out = tf.keras.layers.Dense(num_outputs,name="my_out",activation=None,kernel_initializer=normc_initializer(0.01))(layer_2)
            value_out = tf.keras.layers.Dense(1,name="value_out",activation=tf.nn.softmax,kernel_initializer=normc_initializer(0.01))(layer_2)
            self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
            self.base_model.summary()
            self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"]["real_obs"])
        inf_mask = tf.maximum(tf.math.log(input_dict["obs"]["action_mask"]), tf.float32.min)
        output = model_out+inf_mask
        return output, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

def gen_policy(GENV,i):
    config = {
        "model": {
            "custom_model": 'GomokuModel',
        }
    }
    return (None, GENV.observation_space, GENV.action_space, config)

def map_fn(agent_id):
    if agent_id=='agent_0':
        return 'policy_0'
    else:
        return 'policy_1'