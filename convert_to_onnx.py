import torch
from backbone.model_mobilefacenet import MobileFaceNet

name = "sface-kd-sface-7"

backbone = MobileFaceNet(128)
backbone.load_state_dict(torch.load(f"weights/new/{name}.pth"))
backbone.eval()

image_tensor = torch.randn(1, 3, 112, 112)

onnx_program = torch.onnx.export(
    backbone,
    image_tensor,
    dynamo=True,
    export_params=True,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)
onnx_program.optimize()
onnx_program.save(f"weights/new/{name}.onnx")
