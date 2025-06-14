import cv2
import numpy as np
import mediapipe as mp
import hand_tracking_mod as htm
import os as os
import time as time

folderPath = "painter"
myList = os.listdir(folderPath)
print(myList)

overlaylist = []
for filename in myList:
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(folderPath, filename))
        overlaylist.append(img)
print(len(overlaylist))
header = overlaylist[0]

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
drawColor = (0,0,255)
ctime = 0
ptime = 0

detector = htm.HandDetector(detectioncon = 0.8)

xp, yp = 0, 0
brushThickness = 20
eraserThickness = 120
imgCanvas = np.zeros((720,1280,3) , np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    Canvas = cv2.flip(imgCanvas,1)
    lmlist = detector.findposition(img, draw=False)

    if len(lmlist) !=0 :
        #print(lmlist)
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers = detector.fingersup()
        print(fingers)

        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1, y1), (x2, y2), drawColor, 2, cv2.FILLED)
            print("Selection mode")

        if(y1 < 125):

            if 0<x1<124:
                header = overlaylist[0]
                drawColor = (0,0,255)
            elif 345<x1<589:
                header = overlaylist[1]
                drawColor = (0, 255, 0)
            elif 819<x1<1054:
                header = overlaylist[2]
                drawColor = (255, 0, 0)
            elif 1054<x1:
                header = overlaylist[3]
                drawColor = (0, 0, 0)

        if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 10, drawColor, 2)
                print("Drawing Mode")

                if xp == 0 and yp == 0:
                    xp,yp = x1,y1

                if drawColor == (0,0,0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp,yp = x1,y1

    header = cv2.resize(header, (img.shape[1], 125))
    img[0:125, 0:img.shape[1]] = header

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", Canvas)

    cv2.waitKey(1)