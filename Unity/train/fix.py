import numpy as np

import onnx
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


def add_constant(model, input_name, output_name: str, dims, vals):
    inp = onnx.helper.make_tensor(input_name, onnx.TensorProto.FLOAT, dims, vals)
    model.graph.initializer.append(inp)
    output = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape=[])
    proto_model.graph.output.append(output)
    node = onnx.helper.make_node('Identity', inputs=[input_name], outputs=[output_name])
    model.graph.node.append(node)


graph = gs.import_onnx(onnx.load('ray_torch/model.onnx'))
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
proto_model.graph.node.append(onnx.helper.make_node(name='n1', op_type='ArgMax', inputs=["logits"],
                                                    outputs=["discrete_actions"], axis=1, keepdims=False))
actions_info = onnx.helper.make_tensor_value_info("discrete_actions", onnx.TensorProto.INT64, shape=["batch_size", 1])
proto_model.graph.output.append(actions_info)

add_constant(proto_model, "version_number.1", "version_number", [1], [3])
add_constant(proto_model, "memory_size_vector", "memory_size", [1], [0])
add_constant(proto_model, "discrete_act_size_vector", "discrete_action_output_shape", [1, 1], [3])


onnx.checker.check_model(proto_model)
onnx.save(proto_model, 'ray_torch/model_fixed.onnx')
