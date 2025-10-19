import cv2
from pathlib import Path
import numpy as np


path_to_video = Path(r"C:\Users\aaron\freemocap_data\recording_sessions\session_2025-05-22_13_36_08\recording_13_43_07_gmt-4_calibration\synchronized_videos\Camera_001_synchronized.mp4")

## 1) Make a charuco board object
squares_x = 7
squares_y = 5
square_size = 1 #this is the size of the black square in meters

marker_size = square_size * 0.8 #this is the recommended ratio between square and marker size

default_dict_id = cv2.aruco.DICT_4X4_250
aruco_dict = cv2.aruco.getPredefinedDictionary(default_dict_id)

board = cv2.aruco.CharucoBoard(
    size = (squares_x, squares_y),
    squareLength = square_size,
    markerLength = marker_size,
    dictionary = aruco_dict
)

detector = cv2.aruco.CharucoDetector(board)

def process_image(image: np.ndarray, detector: cv2.aruco.CharucoDetector):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray_image)

    return charuco_corners, charuco_ids, marker_corners, marker_ids
cap = cv2.VideoCapture(str(path_to_video))

charuco_corners_all = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    charuco_corners, charuco_ids, marker_corners, marker_ids = process_image(frame, detector)

    charuco_corners_all.append(charuco_corners)

cap.release()

f = 2