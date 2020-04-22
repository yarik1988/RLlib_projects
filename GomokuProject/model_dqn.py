from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import *
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
import numpy as np
import aux_fn

BOARD_SIZE = 6
NUM_IN_A_ROW = 4


class GomokuModel(DistributionalQTFModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kw):
        super(GomokuModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

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
            kz = [4, 4, 4, 4]
            filt = [8, 8, 8, 1]
            regul = tf.keras.regularizers.l2(self.model_config['custom_options']['reg_loss'])
            for i in range(len(kz)):
                cur_layer = tf.keras.layers.Conv2D(kernel_size=kz[i], filters=filt[i], padding='same',
                                                   kernel_regularizer=regul, activation='elu', name="Conv_" + str(i))(cur_layer)
            self.base_model = tf.keras.Model(self.inputs, cur_layer, name='DQN_model')
            self.register_variables(self.base_model.variables)

    def get_sym_output(self, board, rotc=0, is_flip=False):
        board_tmp = tf.identity(board)
        board_tmp = tf.image.rot90(board_tmp, k=rotc % 4)
        if is_flip:
            board_tmp = tf.image.flip_up_down(board_tmp)
        model_out = self.base_model(board_tmp)

        if is_flip:
            model_out = tf.image.flip_up_down(model_out)

        model_out = tf.image.rot90(model_out, k=(4-rotc) % 4)
        model_out_fin = tf.reshape(model_out, [-1, self.outputs**2])
        return model_out_fin

    def forward(self, input_dict, state, seq_lens):
        self.action_mask = input_dict["obs"]["action_mask"]
        board = input_dict["obs"]["real_obs"]
        if self.use_symmetry:
            model_rot_out = [None]*8
            for j in range(2):
                for i in range(4):
                    model_rot_out[j*4+i] = self.get_sym_output(board, i, j)
            model_out = tf.math.add_n(model_rot_out)
        else:
            model_out = self.base_model(board)
            model_out = tf.reshape(model_out, [-1, self.outputs ** 2])
        return model_out, state

    def value_function(self):
        return self.value_out

    def get_q_value_distributions(self, model_out):
        model_out, logits, dist = self.q_value_head(model_out)
        inf_mask = tf.maximum(tf.log(self.action_mask), tf.float32.min)
        return model_out+inf_mask, logits, dist

def gen_policy(GENV, i):
    ModelCatalog.register_custom_model("GomokuModel_{}".format(i), GomokuModel)
    config = {
        "model": {
            "custom_model": "GomokuModel_{}".format(i),
            "custom_options": {"use_symmetry": True, "reg_loss": 0},
            "fcnet_hiddens": [36],
        },
    }
    return (None, GENV.observation_space, GENV.action_space, config)


def map_fn(np):
    if np == 1:
        return lambda agent_id: "policy_0"
    else:
        return lambda agent_id: "policy_0" if agent_id == 'agent_0' else "policy_1"


def get_trainer(GENV, np):
    trainer = ray.rllib.agents.dqn.ApexTrainer(env="GomokuEnv", config={
        "multiagent": {
            "policies": {"policy_{}".format(i): gen_policy(GENV, i) for i in range(np)},
            "policy_mapping_fn": map_fn(np),
            },
        "num_workers": 2,
        "hiddens": [36],
        "callbacks":
            {"on_episode_end": aux_fn.clb_episode_end},
    })
    print(trainer.get_policy("policy_0").model.base_model.summary())
    qmodel = trainer.get_policy("policy_0").model.q_value_head
    plot_model(qmodel, show_shapes=True, show_layer_names=True)
    return trainer
