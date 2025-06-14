import cv2
import numpy as np
import mediapipe as mp
import time as time

cap = cv2.VideoCapture(1)

mpface = mp.solutions.face_mesh
face = mpface.FaceMesh(max_num_faces=2)
mpdraw = mp.solutions.drawing_utils
drawspec = mpdraw.DrawingSpec(circle_radius=1, thickness=1)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)

    if results.multi_face_landmarks :
        for faceLms in results.multi_face_landmarks :
            mpdraw.draw_landmarks(img, faceLms, mpface.FACEMESH_CONTOURS, drawspec, drawspec)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
