<details open>
<summary>Install</summary>

Pip install the ultralytics package including all requirements.txt in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

### Python

MESG-CDFA may be used directly in a Python environment:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")

# Train the model
train_results = model.train(
    data="SSDD.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
```
