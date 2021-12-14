##########################################################
# lane_detector.py
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


class LaneDetector:
    def __init__(self, is_video=False, width=1280, height=720):
        self.vertices = None
        self.is_video = is_video
        self.width = width
        self.height = height
        self.frame = None

    """  General methods for setting files and getting information """

    def set_next_frame(self, frame):
        self.frame = frame

    def set_vertices(self, vertices):
        self.vertices = vertices

    """ Equalize histograms """

    def equalize_gray_image(self, frame):
        return cv2.equalizeHist(frame)

    def equalize_hls_image(self, frame):
        hue, light, sat = cv2.split(frame)
        light = cv2.equalizeHist(light)
        return cv2.merge([hue, light, sat])

