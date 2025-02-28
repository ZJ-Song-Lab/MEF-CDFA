import torch
from models.yolo import Model

# 加载模型
try:
    model = Model('ultralytics/cfg/models/11/yolo11-CDFA-Edge_4.yaml')
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# 打印模型结构
print(model)

# 假设输入尺寸为 [1, 3, 640, 640]
x = torch.randn(1, 3, 640, 640)

# 前向传播并打印每一层的输出尺寸
try:
    for i, layer in enumerate(model.model):
        x = layer(x)
        print(f"Layer {i}: {x.shape}")
except Exception as e:
    print(f"前向传播出错: {e}")