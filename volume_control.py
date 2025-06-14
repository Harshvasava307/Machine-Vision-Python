import math
import numpy as np
import cv2
import mediapipe as mp
import time as time
import hand_tracking_mod as htm

from ctypes import cast, POINTER
from comtypes import CLSCTX, CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume

wCam, hCam = 640, 480
ctime = 0
ptime = 0

detector = htm.HandDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMasterVolumeLevel()
volrange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)

minvol = volrange[0]
maxvol = volrange[1]


cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)


while True:
    success, img = cap.read()
    img  = detector.findHands(img)
    lmList = detector.findposition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (int(x1), int(y1)), 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (int(x2), int(y2)), 5, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        print(length)

        #handrange = 50 - 100
        #volumerange = -65 -0

        vol = np.interp(length, [50, 220], [minvol, maxvol])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (int(x1), int(y1)), 5, (255, 0, 255), cv2.FILLED)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2 )
    cv2.imshow('Video', img)
    cv2.waitKey(1)
