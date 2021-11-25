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

    """ Edge detection """

    #
    # Perform Edge detection with Sobel filter
    def apply_sobel_edge_detection(self, frame):
        # For x direction
        sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
        # For y direction
        sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
        # Calculate filtered matrix
        grad = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return (grad * 255 / grad.max()).astype(np.uint8)

    def apply_canny_edge_det(self, frame, thresholds=(50, 150)):
        return cv2.Canny(frame, thresholds[0], thresholds[1])

    """ Conversion of color space """

    def bgr_to_hls(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    def bgr_to_gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

    def __create_hls_street_mask(self, frame):
        avg_0, avg_1, avg_2 = self.__get_top_avg_val(frame)  # (14.75, 166.23, 53.9)
        lower = (avg_0 * 0.33, avg_1 * 0.66, avg_2 * 0.75)  # (5, 110, 40)
        upper = (avg_0 * 1.8, avg_1 * 1.2, avg_2 * 1.3)  # (25, 200, 70)
        return cv2.inRange(frame, lower, upper)

    def __get_top_street_limit(self, frame):
        offset_street = 0.8
        for off in range(0, 8):
            if np.mean(frame[int(self.height * (off / 10)):int(self.height * ((off + 1) / 10)), :, :]) > 10:
                offset_street = off / 10
                break
        return offset_street

    def set_top_roi(self, hls_frame, bgr_frame):
        mask_street = self.__create_hls_street_mask(hls_frame)
        street_image = cv2.bitwise_and(bgr_frame, bgr_frame, mask=mask_street)

        offset_street = self.__get_top_street_limit(street_image)
        return offset_street

    def set_bottom_roi(self, frame):
        # Mask bottom of image (10%)
        frame[int(self.height * 0.9):, :, :] = (0, 0, 0)

    """  Blur frames """

    def apply_gaussian_blur(self, image, kernel=(15, 15)):
        return cv2.GaussianBlur(image, kernel, 0)

    """  Detect lanes on frame """

    def create_hls_lane_mask(self, hls_frame):
        # White color mask
        lower = np.uint8([0, 210, 0])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(hls_frame, lower, upper)
        # yellow color mask
        lower = np.uint8([10, 0, 110])
        upper = np.uint8([40, 255, 255])
        yellow_mask = cv2.inRange(hls_frame, lower, upper)
        # combine the mask
        return cv2.bitwise_or(white_mask, yellow_mask)

    def apply_mask(self, frame, mask):
        return cv2.bitwise_and(frame, frame, mask=mask)

    # 80, 255 for sat and 120, 255 for red
    def calculate_threshold(self, frame, thresholds):
        return cv2.threshold(frame, thresholds[0], thresholds[1], cv2.THRESH_BINARY)

    def create_binary_image(self, frame_thresh_1, frame_thresh_2):
        return cv2.bitwise_and(frame_thresh_1, frame_thresh_2)

    def create_binary_lane_mask(self, hls_frame):
        _, saturation_threshold = self.calculate_threshold(hls_frame[:, :, 1], (80, 255))
        _, red_threshold = self.calculate_threshold(hls_frame[:, :, 2], (120, 255))
        return self.create_binary_image(saturation_threshold, red_threshold)

    """  Show edited frames """

    def show_hls_channels(self, hls_frame):
        cv2.imshow('hue', hls_frame[:, :, 0])
        cv2.imshow('lightness', hls_frame[:, :, 1])
        cv2.imshow('saturation', hls_frame[:, :, 2])
