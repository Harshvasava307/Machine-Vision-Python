import numpy as np
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mppose = mp.solutions.pose
pose = mppose.Pose()
mpdraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks :
            mpdraw.draw_landmarks(img, results.pose_landmarks, mppose.POSE_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
