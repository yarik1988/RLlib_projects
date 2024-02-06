import os

import onnx
import torch
from torch import nn
from typing import NamedTuple, List
from torch.nn import Parameter
from onnx2torch import convert
import onnx_graphsurgeon as gs
@gs.Graph.register()
def replace_with_concat(self, inputs, outputs):
    for inp in inputs:
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="Concat", inputs=inputs, outputs=outputs, attrs={"axis": 1})

class WrapperModel(torch.nn.Module):
    def __init__(
            self,
            ray_torchmodel,
            discrete_output_sizes
    ):
        """
        Wraps the VisualQNetwork adding extra constants and dummy mask inputs
        required by runtime inference with Sentis.

        For environment continuous actions outputs would need to add them
        similarly to how discrete action outputs work, both in the wrapper
        and in the ONNX output_names / dynamic_axes.
        """
        super(WrapperModel, self).__init__()
        self.ray_torchmodel = ray_torchmodel

        # version_number
        #   MLAgents1_0 = 2   (not covered by this example)
        #   MLAgents2_0 = 3
        version_number = torch.Tensor([3])
        self.version_number = Parameter(version_number, requires_grad=False)

        # memory_size
        # TODO: document case where memory is not zero.
        memory_size = torch.Tensor([0])
        self.memory_size = Parameter(memory_size, requires_grad=False)

        # discrete_action_output_shape
        output_shape=torch.Tensor([discrete_output_sizes])
        self.discrete_shape = Parameter(output_shape, requires_grad=False)


    # if you have discrete actions ML-agents expects corresponding a mask
    # tensor with the same shape to exist as input
    def forward(self, obs_0: torch.tensor, mask: torch.tensor):
        model_result = self.ray_torchmodel(obs_0)
        logits = model_result
        inf_mask = torch.clamp(torch.log(mask), min=-3.4e38)
        masked_logits = logits + inf_mask
        m = nn.Softmax(dim=-1)
        probs = m(masked_logits)
        det_action = torch.argmax(probs, dim=1, keepdim=True)
        prob_action=torch.multinomial(probs,1)
        return [prob_action],[det_action], self.discrete_shape, self.version_number, self.memory_size

def fix_graph(path):
    path_head,path_tail=os.path.split(path)
    filename_no_ext, _ =os.path.splitext(path_tail)
    out_path=os.path.join(path_head,filename_no_ext+"_wrapped.onnx")
    graph = gs.import_onnx(onnx.load(path))
    graph.inputs.pop()
    graph.outputs.pop()
    graph.inputs[0].name = 'obs_0'  # rename input
    graph.outputs[0].name = 'logits'  # rename input
    tmap = graph.tensors()
    inputs = [tmap["obs_0"]]
    for ten in tmap:
        if "Gemm" in ten:
            outputs = [tmap[ten]]
            break

    graph.replace_with_concat(inputs, outputs)
    graph.cleanup().toposort()
    proto_model = gs.export_onnx(graph)
    ray_torchmodel = convert(proto_model)
    num_actions = 3
    wrapped_model = WrapperModel(ray_torchmodel, [num_actions])
    x1 = torch.randn(1, 4, requires_grad=True)
    mask = torch.ones(1, num_actions)
    res=wrapped_model(x1,mask)

    torch.onnx.export(
        wrapped_model,
        (x1, mask),
        out_path,
        export_params=True,
        opset_version=9,
        # input_names must correspond to the WrapperNet forward parameters
        # obs will be obs_0, obs_1, etc.
        input_names=["obs_0", "action_masks"],
        # output_names must correspond to the return tuple of the WrapperNet
        # forward function.
        output_names=["discrete_actions","deterministic_discrete_actions", "discrete_action_output_shape",
                      "version_number", "memory_size"],
        # All inputs and outputs should have their 0th dimension be designated
        # as 'batch'
        dynamic_axes={'obs_0': {0: 'batch'},
                      'action_masks': {0: 'batch'},
                      'discrete_actions': {0: 'batch'},
                      'deterministic_discrete_actions': {0: 'batch'},
                      'discrete_action_output_shape': {0: 'batch'}
                     }
        )

if __name__ == "__main__":
    fix_graph("ray_torch/model.onnx")