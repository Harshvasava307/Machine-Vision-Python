import cv2
import numpy as np
import mediapipe as mp
import time as time
import hand_tracking_mod as htm


cap = cv2.VideoCapture(1)
wcam, hcam = 640,480
cap.set(3, wcam)
cap.set(4, hcam)

cTime = 0
pTime = 0

detector = htm.HandDetector()


while True:
    success, img = cap.read()
    img  = detector.findHands(img)
    lmList = detector.findposition(img, draw=False)

    if len(lmList) != 0:
        if lmList[8][2] < lmList[6][2]:
                cv2.putText(img, str(1),(20,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2 )

    cv2.imshow('frame', img)
    cv2.waitKey(1)