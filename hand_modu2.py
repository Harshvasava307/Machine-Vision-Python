import cv2
import numpy as np
import mediapipe as mp
import time as time
import math

class HandDetector():
    def __init__(self, mode=False, max_num_hands=2, detectioncon=0.7, trackcon=0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=self.detectioncon,
                                         min_tracking_confidence=self.trackcon)
        self.mpdraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # List of tip finger IDs

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def findposition(self, img, hand=0, draw=True):
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[hand]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])  # Store [ID, x, y] for each landmark
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.lmlist

    def fingersup(self):
        fingers = []
        if len(self.lmlist) > 0:
            # Thumb: Compare x-coordinates to check if the thumb is up
            if self.lmlist[self.tipIds[0]][1] < self.lmlist[self.tipIds[0] - 1][1]:
                fingers.append(1)  # Thumb is open
            else:
                fingers.append(0)  # Thumb is closed

            # Other fingers: Compare y-coordinates to check if the fingers are up
            for id in range(1, 5):
                if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                    fingers.append(1)  # Finger is open
                else:
                    fingers.append(0)  # Finger is closed
        else:
            fingers = [0, 0, 0, 0, 0]  # Return all 0 if no hand is detected

        return fingers

    def findDistance(self, p1, p2, img=None):
        if isinstance(p1, tuple) and len(p1) == 2 and isinstance(p2, tuple) and len(p2) == 2:
            x1, y1 = p1
            x2, y2 = p2
            # Euclidean distance formula
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if img is not None:
                # Optionally, draw a line between the points on the image
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)

            return distance, img  # Returning the distance along with the image
        else:
            raise ValueError("p1 and p2 must be tuples of (x, y) coordinates")


def main():
    cTime = 0
    pTime = 0

    cap = cv2.VideoCapture(1)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findposition(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        if len(lmlist) != 0:
            print(lmlist[4])

        cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.imshow('Video', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()