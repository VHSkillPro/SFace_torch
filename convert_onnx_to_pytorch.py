import torch
from onnx2torch import convert

from backbone.model_mobilefacenet import MobileFaceNet

onnx_model_path = "weights/face_recognition_sface_2021dec.onnx"
torch_model = convert(onnx_model_path, attach_onnx_mapping=True)

# model = MobileFaceNet(128)
# print(*model.state_dict().keys(), sep="\n")

print(torch_model)
# print(*torch_model.state_dict().keys(), sep="\n")
# print(torch_model.state_dict()["initializers.onnx_initializer_2"])
# torch.save(torch_model.state_dict(), "weights/face_recognition_sface_2021dec.pth")
