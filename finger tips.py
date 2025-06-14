import cv2
import numpy as np
import mediapipe as mp
import time as time

cap = cv2.VideoCapture(1)

ctime = 0
ptime = 0

mphands = mp.solutions.hands
hand = mphands.Hands()
mpdraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hand.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

            for id,lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                cv2.circle(img, (cx,cy), 5, (255, 0, 255), cv2.FILLED)

            fingers = [handLms.landmark[i].y < handLms.landmark[i - 2].y for i in [4, 8, 12, 16, 20]]
            count = sum(fingers)
            cv2.putText(img, f"Fingers: {count}", (40, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,str(int(fps)),(20,70),cv2.FONT_HERSHEY_SIMPLEX,0,(255,255,255),2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)