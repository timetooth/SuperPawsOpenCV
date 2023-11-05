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
out = cv2.VideoWriter(video_path_out,cv2.VideoWriter_fourcc(*'mp4v'),int(cap.get(cv2.CAP_PROP_FPS)),(W,H))

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
        mask = np.zeros_like()
        print("the coordinates are\n",x1,x2,y1,y2)
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
    # Set the intensity of the cropped region to the original frame
        if score > threshold:
            for y in range(y1,y2):
                for x in range(x1,x2):
                    mask[y,x] = frame[y,x]
        cv2.rectangle(mask, (x1, y1), (x2, y2),(0,255,0),5),
        # if score > threshold:
        #     cv2.putText(mask, x, (int(x1), int(y1 - 10)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(mask)
    ret, mask = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

