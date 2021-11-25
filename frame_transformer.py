##########################################################
# frame_transformer.py
#
# SPDX-FileCopyrightText: Copyright 2021 Fabbio Protopapa
#
# SPDX-License-Identifier: MIT
#
# Wrapper for frame transformation functions
#
# ########################################################
#
# Import libraries
import cv2
import numpy as np

cv2ColorSpace = {
    'hls': cv2.COLOR_BGR2HLS,
    'hsv': cv2.COLOR_BGR2HSV,
    'luv': cv2.COLOR_BGR2LUV,
    'yuv': cv2.COLOR_BGR2YUV,
    'lab': cv2.COLOR_BGR2LAB,
    'gray': cv2.COLOR_BGR2GRAY,
}

class FrameTransformer:
    @staticmethod
    def bgr_to_x(frame, colorSpace):
        return cv2.cvtColor(frame, cv2ColorSpace[colorSpace])

    @staticmethod
    def apply_gaussian_blur(frame, kernel=(5, 5)):
        return cv2.GaussianBlur(frame, kernel, 0)

    @staticmethod
    def apply_sobel_edge_detection(frame):
        # For x direction
        sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
        # For y direction
        sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
        # Calculate filtered matrix
        grad = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return (grad * 255 / grad.max()).astype(np.uint8)

    @staticmethod 
    def apply_canny_edge_det(frame, thresholds=(30, 150)):
        return cv2.Canny(frame, thresholds[0], thresholds[1])

    @staticmethod
    def create_channel_mask(channel, threshold):
        return cv2.inRange(channel, threshold[0], threshold[1])

    @staticmethod
    def apply_threshold(channel, threshold=(125, 255), thresh_typ=cv2.THRESH_BINARY):
        return cv2.threshold(channel, threshold[0], threshold[1], thresh_typ)

    @staticmethod
    def combine_mask(channel, mask):
        return cv2.bitwise_or(channel, channel, mask=mask)

    @staticmethod
    def intersect_mask(channel, mask):
        return cv2.bitwise_and(channel, channel, mask=mask)


def main():
    Transformer = FrameTransformer()

#
# Run as script
if __name__ == "__main__":
    main()