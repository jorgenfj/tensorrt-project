import torch
from yolo.python.unet.model.model import UNet

data = torch.zeros(1, 3, 640, 640).cuda()

model = UNet(n_classes=1)
model.eval()

torch.onnx.export(UNet, data, 'unet.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=13
)