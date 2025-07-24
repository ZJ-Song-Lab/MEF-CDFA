<details open>
<summary>Install</summary>

Pip install the ultralytics package including all requirements.txt in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

### Python

MEF-CDFA may be used directly in a Python environment:

```python

# Load a model
model = YOLO("SOTA/weights/best.pt")

# Train the model
train_results = model.train(
    data="SSDD.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="0", 
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model

import os

# 图片路径
image_paths = ["C:/Users/宋子京/Desktop/RT论文1/图/MSEIS.jpg"]
# 保存结果的文件夹（不存在则创建）
output_dir = "detection_results"
os.makedirs(output_dir, exist_ok=True)

for path in image_paths:
    # 检测图像
    results = model(path)
    # 获取原图片文件名（用于命名结果文件）
    img_name = os.path.basename(path)
    # 保存路径
    save_path = os.path.join(output_dir, f"detected_{img_name}")
    # 保存结果
    results[0].save(save_path)
    print(f"检测结果已保存至：{save_path}")
```
