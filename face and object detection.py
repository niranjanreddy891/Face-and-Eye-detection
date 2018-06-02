####  @ author: niranjanreddy891@gmail.com

import numpy as np
import cv2

# https://github.com/niranjanreddy891/Face-and-Eye-detection

# https://github.com/niranjanreddy891/Face-and-Eye-detection/blob/master/face_detect.xml
face_cascade = cv2.CascadeClassifier('E:/Face and object detection/face_detection.xml')
# https://github.com/niranjanreddy891/Face-and-Eye-detection/blob/master/eye_detection.xml
eye_cascade = cv2.CascadeClassifier('E:/Face and object detection/eye_detection.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('Face and Eye detection', img)
    k = cv2.waitKey(50) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
