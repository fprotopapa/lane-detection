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
# Print Image Information, return image resolution (width, height, channel)
def get_image_information(image, filename):
    height, width, channel = image.shape
    print("""
    File:     {}
    Channels: {}
    Width:    {}
    Height:   {}
    """.format(
        filename,
        channel,
        width,
        height
    ))
    return width, height, channel


#
# Perform Edge detection with Sobel filter
def get_sobel_edge_detection(frame):
    # For x direction
    sobel_x = np.absolute(cv2.Sobel(frame, cv2.CV_64F, 1, 0, 3))
    # For y direction
    sobel_y = np.absolute(cv2.Sobel(frame, cv2.CV_64F, 0, 1, 3))
    # Calculate filtered matrix
    return np.sqrt(sobel_x ** 2 + sobel_y ** 2)


def __get_top_avg_val(image, width, height):
    width_low = int(width / 2 - width * 0.2)
    width_upp = int(width / 2 + width * 0.2)
    height_low = int(height * 0.7)
    height_upp = int(height * 0.85)
    return (
        np.mean(image[height_low:height_upp, width_low:width_upp, 0]),
        np.mean(image[height_low:height_upp, width_low:width_upp, 1]),
        np.mean(image[height_low:height_upp, width_low:width_upp, 2])
    )


def __create_hls_street_mask(image, width, height):
    avg_0, avg_1, avg_2 = __get_top_avg_val(image, width, height)  # (14.75, 166.23, 53.9)
    lower = (avg_0 * 0.33, avg_1 * 0.66, avg_2 * 0.75)  # (5, 110, 40)
    upper = (avg_0 * 1.8, avg_1 * 1.2, avg_2 * 1.3)  # (25, 200, 70)
    return cv2.inRange(image, lower, upper)


def __get_top_street_limit(image, height):
    offset_street = 0.8
    for off in range(0, 8):
        if np.mean(image[int(height * (off / 10)):int(height * ((off + 1) / 10)), :, :]) > 10:
            offset_street = off / 10
            break
    return offset_street


def set_top_roi(hls_image, bgr_image, width, height):
    mask_street = __create_hls_street_mask(hls_image, width, height)
    street_image = cv2.bitwise_and(bgr_image, bgr_image, mask=mask_street)

    offset_street = __get_top_street_limit(street_image, height)
    return offset_street


def set_bottom_roi(image, height):
    # Mask bottom of image (10%)
    image[int(height * 0.9):, :, :] = (0, 0, 0)
