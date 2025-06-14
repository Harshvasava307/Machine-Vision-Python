import time
import numpy as np
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)
mphand = mp.solutions.hands
hands = mphand.Hands()
mpdraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, handLms, mphand.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2 )
    cv2.imshow("Image", img)
    cv2.waitKey(1)