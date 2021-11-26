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


class LaneDetector:
    def __init__(self, file, is_video=True):
        self.filename = None
        self.is_video = is_video
        self.width = 0
        self.height = 0
        self.channel = None
        self.video = None
        self.frame = None
        if self.is_video:
            self.video = file
            self.get_video_shape()
        else:
            self.frame = file
            self.get_image_shape()

    """  General methods for setting files and getting information """

    def set_next_frame(self, frame):
        self.frame = frame

    def change_file(self, file):
        if self.is_video:
            self.video = file
            self.get_video_shape()
        else:
            self.frame = file
            self.get_image_shape()

    def get_video_shape(self):
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #
    # Print Video Information, return video resolution (width, height)
    def get_video_information(self, filename=None):
        if filename is None:
            filename = "-"
        print("""
        File:     {}
        FPS:      {}
        # Frames: {}
        Width:    {}
        Height:   {}
        """.format(
            filename,
            int(self.video.get(cv2.CAP_PROP_FPS)),
            int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)),
            self.width,
            self.height
        ))
        return self.width, self.height

    def get_image_shape(self):
        self.height, self.width, self.channel = self.frame.shape

    #
    # Print Image Information, return image resolution (width, height, channel)
    def get_image_information(self, filename=None):
        if filename is None:
            filename = "-"
        print("""
        File:     {}
        Channels: {}
        Width:    {}
        Height:   {}
        """.format(
            filename,
            self.channel,
            self.width,
            self.height
        ))
        return self.width, self.height, self.channel

    """ Equalize histograms """

    def equalize_gray_image(self, frame):
        return cv2.equalizeHist(frame)

    def equalize_hls_image(self, frame):
        hue, light, sat = cv2.split(frame)
        light = cv2.equalizeHist(light)
        return cv2.merge([hue, light, sat])

    """ Set ROI """

    def __get_top_avg_val(self, frame):
        width_low = int(self.width / 2 - self.width * 0.2)
        width_upp = int(self.width / 2 + self.width * 0.2)
        height_low = int(self.height * 0.7)
        height_upp = int(self.height * 0.85)
        return (
            np.mean(frame[height_low:height_upp, width_low:width_upp, 0]),
            np.mean(frame[height_low:height_upp, width_low:width_upp, 1]),
            np.mean(frame[height_low:height_upp, width_low:width_upp, 2])
        )
