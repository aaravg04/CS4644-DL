# downloads yolo and runs it
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from ultralytics import YOLO


# load the yolo model
model = YOLO('yolo11m.pt')

# strip last 2 layers of the model

# run the model on the image
results = model("image.png")

# Get the first result
result = results[0]

# Get the original image
image = result.orig_img

# Draw boxes and labels on the image
boxes = result.boxes
for box in boxes:
    # Get coordinates
    x1, y1, x2, y2 = box.xyxy[0]
    
    # Get class name and confidence
    class_id = int(box.cls[0])
    conf = float(box.conf[0])
    class_name = result.names[class_id]
    
    # Draw rectangle
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    # Create label text with class name and confidence
    label = f'{class_name}: {conf:.2f}'
    
    # Put text above the box
    cv2.putText(image, label, (int(x1), int(y1-10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Display the image
cv2.imshow('YOLO Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

