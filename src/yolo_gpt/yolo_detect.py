from ultralytics import YOLO

model = YOLO("yolo11n.pt")
class_names = model.names

def generate_detections(path: str):
    results = model(path)
    result = results[0].boxes

    out = ""
    for i in range(len(result.cls)):
        if result.conf[i].item() < 0.5:  # confidence threshold
            continue
        cls_name = class_names[int(result.cls[i])]
        out += f"{cls_name}, {[round((result.xyxy[i][0].item() + result.xyxy[i][2].item()) / 2, 1), round((result.xyxy[i][1].item() + result.xyxy[i][3].item()) / 2, 1)]}\n"

    return out
