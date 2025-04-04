import cv2
import numpy as np
from ultralytics import YOLO
import cvzone


model = YOLO("yolov10s.pt")


with open("classes.txt", "r") as f:
    class_list = f.read().split("\n")


person_class_id = class_list.index("person") if "person" in class_list else -1

cap = cv2.VideoCapture("fall5.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))


    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes.data:
            x1, y1, x2, y2, conf, class_id = box[:6].tolist()
            class_id = int(class_id)


            if class_id != person_class_id:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


            color = (0, 255, 0)
            label = "person"
            h, w = y2 - y1, x2 - x1
            if h < w:
                label = "person_fall"
                color = (0, 0, 255)


            cvzone.putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=1, colorR=color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
