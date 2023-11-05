import os
from ultralytics import YOLO
import cv2

import numpy as np

video_dir = os.path.join('.','video')

video_path = os.path.join(video_dir,'shorty.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret , frame = cap.read()
H,W,_ = frame.shape
out = cv2.VideoWriter(video_path_out,cv2.VideoWriter_fourcc(*'MP4V'),int(cap.get(cv2.CAP_PROP_FPS)),(W,H))

model_path = os.path.join('.','actuall','hund.pt')

model = YOLO(model_path)

threshold = 0.5
x = "can't recognize"

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        print(class_id)
        if(class_id==0):
            x = "chewy"
        else:
            x = "oscar"

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, x, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

