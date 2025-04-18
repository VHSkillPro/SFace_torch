import torch
from onnx2torch import convert

onnx_model_path = "weights/face_recognition_sface_2021dec.onnx"
torch_model = convert(onnx_model_path)

torch.save(torch_model.state_dict(), "weights/face_recognition_sface_2021dec.pth")
