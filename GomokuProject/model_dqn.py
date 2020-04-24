import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import *
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
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
        self.outputs = num_outputs
        input_shp = obs_space.original_space.spaces['real_obs']
        kz = [4, 4, 4, 4]
        filt = [8, 8, 8, 1]
        n = len(kz)
        self.layers = [0] * (n + 1)
        self.layers[0] = tf.keras.layers.Input(shape=input_shp.shape, name="observations")
        for i in range(len(kz)):
            self.layers[i + 1] = tf.keras.layers.Conv2D(kernel_size=kz[i], filters=filt[i], padding='same',
                                                        activation='elu')(self.layers[i])
        self.base_model = tf.keras.Model(self.layers[0], self.layers[n])
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
        model_out_fin = tf.reshape(model_out, [-1, self.outputs])
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
            model_out = tf.reshape(model_out, [-1, self.outputs])
        return model_out, state

    def value_function(self):
        return self.value_out

    def get_q_value_distributions(self, model_out):
        model_out, logits, dist = self.q_value_head(model_out)
        #inf_mask = tf.maximum(tf.log(self.action_mask), -1000)
        #model_out=model_out+inf_mask
        return model_out, logits, dist

def gen_policy(GENV, i):
    ModelCatalog.register_custom_model("GomokuModel_{}".format(i), GomokuModel)
    config = {
        "model": {
            "custom_model": "GomokuModel_{}".format(i),
            "custom_options": {"use_symmetry": False},
            "fcnet_hiddens": [36],
        },
    }
    return (None, GENV.observation_space, GENV.action_space, config)


def map_fn(np):
    if np == 1:
        return lambda agent_id: "policy_0"
    else:
        return lambda agent_id: "policy_0" if agent_id == 'agent_0' else "policy_1"


def get_trainer(GENV, np, is_training=True):
    trainer = ray.rllib.agents.dqn.ApexTrainer(env="GomokuEnv", config={
        "multiagent": {
            "policies": {"policy_{}".format(i): gen_policy(GENV, i) for i in range(np)},
            "policy_mapping_fn": map_fn(np),
            },
        "explore": is_training,
        "num_gpus": 1,
        "num_workers": 2,
        "hiddens": [36],
        "callbacks": aux_fn.MyCallback
    })
    all_model=trainer.get_policy("policy_0").model
    print(all_model.base_model.summary())
    print(all_model.state_value_head.summary())
    print(all_model.q_value_head.summary())
    return trainer
