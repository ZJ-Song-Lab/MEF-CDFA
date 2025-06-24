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
```
