import cv2
import numpy as np
import mediapipe as mp
import time as time


cap = cv2.VideoCapture(1)

mpface = mp.solutions.face_detection
face = mpface.FaceDetection()
mpdraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    if results.detections :
        for id,detection in enumerate(results.detections) :
            mpdraw.draw_detection(img, detection)



    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

