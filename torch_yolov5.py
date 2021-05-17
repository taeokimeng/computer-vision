import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

# Image
img = './images/sample1.jpg' # 'https://ultralytics.com/images/zidane.jpg'

# Inference
results = model(img)
results.show()  # or .print(), .show(), .save()
