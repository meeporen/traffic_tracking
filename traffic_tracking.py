from ultralytics import YOLO
import cv2
import cvzone
import random
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(trigger_zone_draw) < 2:
            trigger_zone_draw.append((x, y))
        elif len(trigger_zone_no_draw) < 2:
            trigger_zone_no_draw.append((x, y))

def is_point_in_zone(x, y, zone):
    if len(zone) < 2:
        return False
    x1, y1 = zone[0]
    x2, y2 = zone[1]
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
video_path = "C:\\work\\terms_of_reference\\video\\test3.mp4"
cap = cv2.VideoCapture(video_path)

model = YOLO("../Yolo-Weights/yolov8l.pt")

trigger_zone_draw = []
trigger_zone_no_draw = []
color_dict = {}
trajectory_dict = {}

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

    results = model.track(img, persist=True, conf=0.6, classes=[2, 5, 7], iou=0.7, device = "cuda") 

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            obj_id = int(box.id.item()) 

            # Confidence
            conf = box.conf[0].item()

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if obj_id is not None:
                if obj_id not in color_dict:
                    color_dict[obj_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                color = color_dict[obj_id]

                cvzone.cornerRect(img, (x1, y1, w, h), l=9, colorR=color)
                cvzone.putTextRect(img, f"ID: {obj_id}", (max(0, x1), max(35, y1)),
                                   scale=0.6, thickness=1, offset=3, colorR=color)

                center_x, center_y = x1 + w // 2, y1 + h // 2

                if is_point_in_zone(center_x, center_y, trigger_zone_draw):
                    if obj_id not in trajectory_dict:
                        trajectory_dict[obj_id] = []
                    trajectory_dict[obj_id].append((center_x, center_y))
                    if len(trajectory_dict[obj_id]) > 1:
                        for i in range(1, len(trajectory_dict[obj_id])):
                            cv2.line(img, trajectory_dict[obj_id][i - 1], trajectory_dict[obj_id][i], color, 2)

                if is_point_in_zone(center_x, center_y, trigger_zone_no_draw):
                    if obj_id in trajectory_dict:
                        trajectory_dict[obj_id].clear()

    if len(trigger_zone_draw) == 2:
        cv2.rectangle(img, trigger_zone_draw[0], trigger_zone_draw[1], (0, 255, 0), 2)
        cv2.putText(img, "Draw Zone", (trigger_zone_draw[0][0], trigger_zone_draw[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if len(trigger_zone_no_draw) == 2:
        cv2.rectangle(img, trigger_zone_no_draw[0], trigger_zone_no_draw[1], (0, 0, 255), 2)
        cv2.putText(img, "No_Draw Zone", (trigger_zone_no_draw[0][0], trigger_zone_no_draw[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()