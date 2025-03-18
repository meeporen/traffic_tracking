from ultralytics import YOLO
import torch
import cv2
import cvzone
import math
import os
import random 

def is_point_in_zone(x, y, zone):
    if len(zone) < 2:
        return False
    x1, y1 = zone[0]
    x2, y2 = zone[1]
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(trigger_zone_draw) < 2:
            trigger_zone_draw.append((x, y))

video_path = "C:\\work\\terms_of_reference\\video\\test3.mp4"
output_folder = "C:\\work\\terms_of_reference\\saved_images_by_freeze_frame"  

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

model = YOLO("../Yolo-Weights/yolov8s.pt")
color_dict = {}
trajectory_dict = {}
trigger_zone_draw = []
frame_count = random.randint(0, 99999)
save_interval = 15  
classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"]

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

while True:
    success, img = cap.read()
    if not success:
        break
    
    results = model.track(img, persist=True, conf=0.7, classes=[2], iou=0.5, device="cuda") 

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            center_x, center_y = x1 + w // 2, y1 + h // 2

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if is_point_in_zone(center_x, center_y, trigger_zone_draw): 
                cvzone.putTextRect(img, f"{conf}", (max(0, x1), max(35, y1)),
                           scale=0.6, thickness=1, offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)

                if frame_count % save_interval == 0:
                    frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, img)

    if len(trigger_zone_draw) == 2:
        cv2.rectangle(img, trigger_zone_draw[0], trigger_zone_draw[1], (0, 255, 0), 2)
        cv2.putText(img, f"Draw Zone", (trigger_zone_draw[0][0], trigger_zone_draw[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()