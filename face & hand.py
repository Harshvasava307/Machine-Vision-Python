import time as time
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)
mphand = mp.solutions.hands
hands = mphand.Hands()

mpface = mp.solutions.face_mesh
face = mpface.FaceMesh(max_num_faces=2)
mpdraw = mp.solutions.drawing_utils
drawspec = mpdraw.DrawingSpec(circle_radius=1, thickness=1, color=(255,0,255))


cTime = 0
pTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, handLms, mphand.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 4:
                    cv2.putText(img, str(int(1)),(20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                    print(id, cx, cy)

    results2 = face.process(imgRGB)
    if results2.multi_face_landmarks :
        for faceLms in results2.multi_face_landmarks :
            mpdraw.draw_landmarks(img, faceLms, mpface.FACEMESH_CONTOURS, drawspec, drawspec)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
