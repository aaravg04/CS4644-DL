# downloads yolo and runs it
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM

# load the yolo model
yolo_model = YOLO('yolo11m.pt')

# load the LLM model and tokenizer
llm_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)

# run the model on the image
yolo_results = yolo_model("bus.jpg")

# Get the first result
yolo_result = yolo_results[0]

# Get the original image
image = yolo_result.orig_img

# Create a list to store all detections and their properties
detections = []

# Draw boxes and labels on the image
boxes = yolo_result.boxes
for box in boxes:
    # Get coordinates
    x1, y1, x2, y2 = box.xyxy[0]
    
    # Calculate centroid
    centroid_x = (x1 + x2) / 2
    centroid_y = (y1 + y2) / 2
    
    # Get class name and confidence
    class_id = int(box.cls[0])
    conf = float(box.conf[0])
    class_name = yolo_result.names[class_id]
    
    # Store detection info
    detections.append({
        'class': class_name,
        'confidence': conf,
        'centroid': (centroid_x, centroid_y),
        'box': (x1, y1, x2, y2)
    })
    
    # Draw rectangle
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    # Create label text with class name and confidence
    label = f'{class_name}: {conf:.2f}'
    
    # Put text above the box
    cv2.putText(image, label, (int(x1), int(y1-10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw centroid
    cv2.circle(image, (int(centroid_x), int(centroid_y)), 4, (0, 255, 0), -1)

# Create simple prompt for LLM
prompt = "Describe spatial relationships between objects in this scene: "
for det in detections:
    prompt += f"a {det['class']} "

# Generate response from LLM
inputs = tokenizer(prompt, return_tensors="pt")
outputs = llm_model.generate(inputs.input_ids, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nLLM Analysis:")
print(response)

# Display the image
cv2.imshow('YOLO Detection with Centroids', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
