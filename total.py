import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

tracker = Tracker()

cap = cv2.VideoCapture('vidyolov8.mp4') 

area = [(735, 459), (47, 182), (54, 144), (886, 29)]
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)

count = 0
area_c = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    # Resize the frame to a smaller size
    scale_percent = 70  # Adjust this value to resize the window to your desired scale
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    results = model.predict(frame, show=False)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id : 
        x3, y3, x4, y4, id = bbox
        cv2.circle(frame, ((x3 + x4) // 2, (y3 + y4) // 2), 4, (0, 0, 255), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        area_c.add(id)

    count = len(area_c)
    cv2.putText(frame, "Total People: " + str(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()
