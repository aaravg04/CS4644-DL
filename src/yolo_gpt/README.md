# YOLO + LLM Implementation

## Theory
The following are the steps this implemention employs:
1. Run an image through YOLO detection
2. Extract the detected objects and their bounding boxes from YOLO's outputs
3. Format the detected objects and their centroids into a prompt
4. Prompt the Llama 3.2 LLM to generate a brief caption for the image

## Preprequisites
1. Install necessary packages from `requirements.txt`
2. Install Ollama and the Llama 3.2 model
3. Download the `yolo11n.pt` model from the Ultralytics website


## Appendix
Main testing is done in the `yolo_transformer_detect.ipynb` file, which imports from `ollama_gpt.py` (Llama usage) and `yolo_detect.py` (YOLO usage)
