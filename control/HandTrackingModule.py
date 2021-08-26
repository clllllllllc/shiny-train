import cv2
import mediapipe as mp
import numpy as np
import time


class HandDetector:
    """A hand tracking class"""

    def __init__(self,
                 mode: int = False,
                 max_hands: int = 2,
                 detection_con: int = 0.5,
                 track_con: int = 0.5) -> None:
        """
        Initialise the values for hand detection
        Input: mode, int, the mode of detection
               max_hands, int, max number of detected hands
               detection_con, float, the confidence in detection
               track_con, float, the confidence in tracking
        Output: None
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_con, self.track_con)
        self.mpDraw = mp.solutions.drawing_utils

        self.results = None

    def find_hands(self, img: np.ndarray, draw: bool = False) -> np.ndarray:
        """
        A function that find the hands
        Input: img, np.ndarray, an array containing the image
               draw, boolean, draw or not draw the hands detected
        Output: img, np.ndarray, an array containing the modified images

        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img: np.ndarray, hand_no: int = 0, draw: bool = False, raw: bool = True) -> list:
        """
        find the positions of the hand
        Input: img, np.ndarray, an array containing the image
               hand_no, int, the index of the hand that you want the position of
               draw, boolean, draw or not draw the locations
               raw, boolean, output raw value or not
        Output: lm_list, array, an 2d array with all the index of the points and the location fo the point
        """
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for index, lm in enumerate(my_hand.landmark):
                # print(id, lm)

                if raw:
                    lm_list.append([index, lm.x, lm.y, lm.z])
                else:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([index, cx, cy, lm.z])
                    if draw:
                        cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

        return lm_list

    def find_distance(self):
        pass


def main():
    """A testing function for the hand detection class, no input or output"""
    p_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        print(type(img))
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        if len(lm_list) != 0:
            print(lm_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
