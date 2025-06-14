import cv2
import time as time
import numpy as np
import mediapipe as mp
import vehicle as v

cap = cv2.VideoCapture(1)
h, w = 1280,720
cap.set(3, w)
cap.set(4, h)

mpcars = v.VehicleType
cars = mpcars.PASSENGER_CAR


ctime = 0
ptime = 0

while True:
    success, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if cars :
        cv2.putText(frame, "Cars detected", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.imshow('frame', frame)
    cv2.waitKey(1)