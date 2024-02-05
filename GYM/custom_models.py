from functools import reduce
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelV2
from ray.rllib.models.tf import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_tf, try_import_torch
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

class RegularizedTFModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(RegularizedTFModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        num_outputs = action_space.n if num_outputs is None else num_outputs
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

class RegularizedTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(RegularizedTorchModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        num_inputs=reduce(lambda accumulator, current: accumulator * current,obs_space.shape)
        num_outputs = action_space.n if num_outputs is None else num_outputs
        self.common = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.action_model=nn.Sequential(
            nn.Linear(256,num_outputs),
        )
        self.value_model=nn.Sequential(
            nn.Linear(256,1),
        )

    def forward(self, input_dict, state, seq_lens):
        common_out = self.common(input_dict["obs"])
        self._value_out=self.value_model(common_out)
        return self.action_model(common_out), state

    def value_function(self):
        return torch.reshape(self._value_out, [-1])

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        l2_lambda = 0.0001
        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        custom_loss = l2_lambda * l2_reg
        if isinstance(policy_loss, list):
            return [single_loss + custom_loss for single_loss in policy_loss]
        else:
            return policy_loss + custom_loss