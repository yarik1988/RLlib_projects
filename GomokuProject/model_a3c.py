from __future__ import absolute_import, division, print_function, unicode_literals
import ray
import sys
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.tf_action_dist import *
from ray.rllib.utils import try_import_tf
tf=try_import_tf()
import numpy as np
import aux_fn

BOARD_SIZE = 6
NUM_IN_A_ROW = 4

dns=tf.keras.layers.Dense(1, activation=None, name='OutV')

class GomokuModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(GomokuModel, self).__init__(obs_space, action_space,num_outputs, model_config, name)
        if 'use_symmetry' in model_config['custom_options']:
            self.use_symmetry = model_config['custom_options']['use_symmetry']
        else:
            self.use_symmetry = False
        with tf.variable_scope(
                tf.VariableScope(tf.AUTO_REUSE, "shared"),
                reuse=tf.AUTO_REUSE,
                auxiliary_name_scope=False):
            input_shp = obs_space.original_space.spaces['real_obs']
            self.inputs = tf.keras.layers.Input(shape=input_shp.shape, name="observations")
            self.outputs = int(np.sqrt(num_outputs))
            can_move = tf.math.equal(self.inputs, tf.fill(tf.shape(self.inputs), 0.0))
            cur_layer = tf.concat([self.inputs, tf.dtypes.cast(can_move, tf.float32)], axis=3)
            kz = [4, 3, 3]
            filt = [32, 16, 8]
            regul = tf.keras.regularizers.l2(self.model_config['custom_options']['reg_loss'])
            for i in range(len(kz)):
                cur_layer = tf.keras.layers.Conv2D(kernel_size=kz[i], filters=filt[i], padding='same',
                                                   kernel_regularizer=regul, name="Conv_" + str(i))(cur_layer)
                #cur_layer = tf.keras.layers.BatchNormalization(name="Batch_" + str(i))(cur_layer)
                cur_layer = tf.keras.layers.Activation(tf.nn.sigmoid, name="Act_" + str(i))(cur_layer)

            layer_out = tf.keras.layers.Conv2D(kernel_size=3, filters=1, padding='same',
                                               kernel_regularizer=regul, name='Out', activation=tf.nn.sigmoid)(cur_layer)
            layer_flat = tf.keras.layers.Flatten(name='FlatFin')(cur_layer)
            value_out = tf.keras.layers.Dense(1, activation=None, kernel_regularizer=regul, name='OutV')(layer_flat)
            self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
            self.register_variables(self.base_model.variables)

    def get_sym_output(self, board, rotc=0, is_flip=False):
        board_tmp = tf.identity(board)
        board_tmp = tf.image.rot90(board_tmp, k=rotc % 4)
        if is_flip:
            board_tmp = tf.image.flip_up_down(board_tmp)
        model_out, val_out = self.base_model(board_tmp)

        if is_flip:
            model_out = tf.image.flip_up_down(model_out)

        model_out = tf.image.rot90(model_out, k=(4-rotc) % 4)
        model_out_fin = tf.reshape(model_out, [-1, self.outputs**2])
        value_out_fin = tf.reshape(val_out, [-1])
        return model_out_fin, value_out_fin

    def forward(self, input_dict, state, seq_lens):
        board = input_dict["obs"]["real_obs"]
        if self.use_symmetry:
            model_rot_out = [None]*8
            value_out = [None]*8
            for j in range(2):
                for i in range(4):
                    model_rot_out[j*4+i], value_out[j*4+i] = self.get_sym_output(board, i, j)
            model_out = tf.math.add_n(model_rot_out)
            self.value_out = tf.math.add_n(value_out)
        else:
            model_out, self.value_out = self.base_model(board)
            model_out = tf.reshape(model_out, [-1, self.outputs ** 2])
            self.value_out = tf.reshape(self.value_out, [-1])
        #min_out = tf.math.reduce_min(model_out)
        #max_out = tf.math.reduce_max(model_out)
        #model_out = tf.math.divide(tf.math.subtract(model_out, min_out), max_out-min_out)
        inf_mask = tf.maximum(tf.math.log(input_dict["obs"]["action_mask"]), tf.float32.min)
        model_out = model_out+inf_mask
        return model_out, state

    def value_function(self):
        return self.value_out

    def policy_variables(self):
        """Return the list of variables for the policy net."""
        return list(self.action_net.variables)

def gen_policy(GENV, i):
    config = {
        "model": {
            "custom_model": 'GomokuModel',
            "custom_options": {"use_symmetry": True, "reg_loss": 0},
        },
    }
    return (None, GENV.observation_space, GENV.action_space, config)


def map_fn(np):
    if np == 1:
        return lambda agent_id: "policy_0"
    else:
        return lambda agent_id: "policy_0" if agent_id == 'agent_0' else "policy_1"


def get_trainer(GENV, np):
    ModelCatalog.register_custom_model("GomokuModel", GomokuModel)
    trainer = ray.rllib.agents.a3c.A3CTrainer(env="GomokuEnv", config={
        "multiagent": {
            "policies": {"policy_{}".format(i): gen_policy(GENV,i) for i in range(np)},
            "policy_mapping_fn": map_fn(np),
            },
        "lr":0.001,
        "callbacks":
            {"on_episode_end": aux_fn.clb_episode_end},
    })
    return trainer


