import autopy
import cv2
import numpy as np
import mediapipe as mp
import time as time
import hand_tracking_mod as htm
import autopy as ap
import os
import math


# Initialize camera and screen resolution
cap = cv2.VideoCapture(1)
detector = htm.HandDetector(max_num_hands=1)
wScr, hScr = autopy.screen.size()
smoothtening = 7
plocx, plocy = 0, 0
clocx, clocy = 0, 0

h, w = 480, 640
cap.set(3, w)
cap.set(4, h)
frameR = 100  # Frame region to define the area for movement

cTime = 0
pTime = 0

while True:
    # Step 1: Find Hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findposition(img, draw=False)

    # Step 2: Get Tip of Middle Finger and Index Finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index Finger Tip (x1, y1)
        x2, y2 = lmList[12][1:]  # Middle Finger Tip (x2, y2)
        print(x1, y1, x2, y2)

    # Step 3: Check which fingers are up
    fingers = detector.fingersup()

    # Step 4: Only Index Finger is up (moving mode)
    if fingers[1] == 1 and fingers[2] == 0:
        # Step 5: Convert Co-ordinates
        x3 = np.interp(x1, (frameR, w - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, h - frameR), (0, hScr))  # Corrected y-coordinate mapping

        # Step 6: Smoothen Values (optional)
        clocx = plocx + (x3 - plocx) / smoothtening
        clocy = plocy + (y3 - plocy) / smoothtening

        # Step 7: Move mouse
        cv2.rectangle(img, (frameR, frameR), (w - frameR, h - frameR), (255, 0, 0), 2)
        autopy.mouse.move(wScr - clocx, clocy)  # Move mouse with inverted x-axis for correct direction
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), 2)
        plocx, plocy = clocx, clocy

    # Step 8: Both Index and Middle Fingers are up (clicking mode)
    if fingers[1] == 1 and fingers[2] == 1:
        # Step 9: Find distance between the fingers
        length, img = detector.findDistance((x1, y1), (x2, y2), img)
        print("Distance:", length)

        # Step 10: Click if distance is small (simulating mouse click)
        if length < 30:  # Adjust distance threshold for click detection
            autopy.mouse.click()
            



    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS
    cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

