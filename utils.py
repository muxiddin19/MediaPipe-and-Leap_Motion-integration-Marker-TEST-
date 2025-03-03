# utils.py
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

def initialize_kalman():
    kf = KalmanFilter(dim_x=3, dim_z=3)
    kf.x = np.array([0., 0., 0.])
    kf.F = np.eye(3)
    kf.H = np.eye(3)
    kf.P *= 1000.
    kf.R = 5
    kf.Q = 0.1
    return kf

def detect_stickers(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])  # Adjust for your sticker color
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sticker_positions = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                sticker_positions.append((cx, cy))
                cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    return sticker_positions[:5]
