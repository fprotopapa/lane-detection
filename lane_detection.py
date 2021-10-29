##########################################################
# lane_detection.py
#
# SPDX-FileCopyrightText: Copyright 2021 Fabbio Protopapa
#
# SPDX-License-Identifier: MIT
#
# Lane detection techniques
#
# ########################################################
#
# Import libraries
import cv2
import numpy as np


#
# Print Video Information, return video resolution (width, height)
def get_video_information(cap, filename):
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("""
    File:     {}
    FPS:      {}
    # Frames: {}
    Width:    {}
    Height:   {}
    """.format(
        filename,
        int(cap.get(cv2.CAP_PROP_FPS)),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width,
        height
    ))
    return width, height


#
# Calculate passed time between two operations per frame
def get_passed_time(start_time, end_time, frames=1):
    return ((end_time - start_time) / cv2.getTickFrequency()) / frames


#
# Perform Edge detection with Sobel filter
def get_sobel_edge_detection(frame):
    # For x direction
    sobel_x = np.absolute(cv2.Sobel(frame, cv2.CV_64F, 1, 0, 3))
    # For y direction
    sobel_y = np.absolute(cv2.Sobel(frame, cv2.CV_64F, 0, 1, 3))
    # Calculate filtered matrix
    return np.sqrt(sobel_x ** 2 + sobel_y ** 2)
