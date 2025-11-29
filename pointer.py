import autopy
import cv2
import numpy as np
import mediapipe as mp
import time as time
import hand_tracking_mod as htm
import autopy as ap
import os
import math
import pyautogui   # Added for drag/drawing


# Initialize camera and screen resolution
cap = cv2.VideoCapture(0)
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

drawing = False   # This stores drag (drawing) state


while True:
    # Step 1: Find Hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findposition(img, draw=False)

    # Step 2: Get Tip of Middle Finger and Index Finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]   # Index finger
        x2, y2 = lmList[12][1:]  # Middle finger

    # Step 3: Check which fingers are up
    fingers = detector.fingersup()

    # ----------------------------------------------------------
    # 1 FINGER = MOVE MOUSE
    # ----------------------------------------------------------
    if fingers[1] == 1 and fingers[2] == 0:

        # If drawing was active → stop drawing
        if drawing:
            pyautogui.mouseUp()
            drawing = False

        # Convert coordinates
        x3 = np.interp(x1, (frameR, w - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, h - frameR), (0, hScr))

        # Smooth movement
        clocx = plocx + (x3 - plocx) / smoothtening
        clocy = plocy + (y3 - plocy) / smoothtening

        # FIXED: natural left-right movement
        autopy.mouse.move(clocx, clocy)

        cv2.rectangle(img, (frameR, frameR), (w - frameR, h - frameR), (255, 0, 0), 2)
        cv2.circle(img, (x1, y1), 8, (0, 255, 0), -1)

        plocx, plocy = clocx, clocy

    # ----------------------------------------------------------
    # 2 FINGERS = DRAW / DRAG (Like MS Paint)
    # ----------------------------------------------------------
    if fingers[1] == 1 and fingers[2] == 1:

        # Convert finger coordinates
        x3 = np.interp(x1, (frameR, w - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, h - frameR), (0, hScr))

        clocx = plocx + (x3 - plocx) / smoothtening
        clocy = plocy + (y3 - plocy) / smoothtening

        # Start drawing (press left mouse)
        if not drawing:
            pyautogui.mouseDown()   # hold left button
            drawing = True

        # FIXED: natural left-right movement
        autopy.mouse.move(clocx, clocy)

        plocx, plocy = clocx, clocy

    # ----------------------------------------------------------

    # FPS Counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
