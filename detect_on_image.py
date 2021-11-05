##########################################################
# detect_on_image.py
#
# SPDX-FileCopyrightText: Copyright 2021 Fabbio Protopapa
#
# SPDX-License-Identifier: MIT
#
# Lane detection on images
#
# ########################################################
#
# Import libraries
import cv2
import numpy as np

import lane_detection as det
import utils

#
# User defined configurations
image_folder_name = 'input_image'


def create_hls_lane_mask(hls_image):
    # White color mask
    lower = np.uint8([0, 210, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hls_image, lower, upper)
    # yellow color mask
    lower = np.uint8([10, 0, 110])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(hls_image, lower, upper)
    # combine the mask
    return cv2.bitwise_or(white_mask, yellow_mask)

def bgr_to_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def bgr_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def apply_gaussian_blur(image, kernel = (15, 15)):
    return cv2.GaussianBlur(image, kernel, 0)

def apply_canny_edge_det(image, thresholds = (50, 150)):
    return cv2.Canny(image, thresholds[0], thresholds[1])

# 80, 255 for sat and 120, 255 for red //cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 0)
def calculate_threshold(image, thresholds):
    return cv2.threshold(image, thresholds[0], thresholds[1], cv2.THRESH_BINARY) #  | cv2.THRESH_OTSU

def create_binary_image(image_thresh_1, image_thresh_2):
    return cv2.bitwise_and(image_thresh_1, image_thresh_2)

def equalize_gray_image(image):
    return cv2.equalizeHist(image)

def equalize_hls_image(image):
    hue, light, sat = cv2.split(image)
    light = cv2.equalizeHist(light)
    return cv2.merge([hue, light, sat])

def show_hls_channels(hls_image):
    cv2.imshow('hue', hls_image[:, :, 0])
    cv2.imshow('lightness', hls_image[:, :, 1])
    cv2.imshow('saturation', hls_image[:, :, 2])

def create_binary_lane_mask(hls_image):
    _, saturation_threshold = calculate_threshold(hls_image[:, :, 1], (80, 255))
    _, red_threshold = calculate_threshold(hls_image[:, :, 2], (120, 255))
    return create_binary_image(saturation_threshold, red_threshold)
#
# Main Application
def main():
    # Get input images
    image_files = utils.get_list_of_input_files(image_folder_name, "jpg")
    # Open test files
    # Index: 2, 5, 6
    bgr_image = cv2.imread(image_files[5])
    filename = utils.get_filename(image_files[5])
    width, height, _ = det.get_image_information(bgr_image, filename)
    #
    # Start processing
    #
    # Convert to HLS (Hue, Lightness, Saturation)

    hls_image = bgr_to_hls(bgr_image)
    # Reduce image to ROI
    det.set_bottom_roi(hls_image, height)
    # Mask top of image
    #offset_street = det.set_top_roi(hls_image, bgr_image, width, height)
    #hls_image[:int(height * offset_street), :, :] = (0, 0, 0)
    # Edge detection
    edge_image = apply_gaussian_blur(hls_image, (15, 15))
    edge_image = det.apply_sobel_edge_detection(edge_image)
    # Use lightness and saturation channels for detecting lanes
    rs_binary = create_binary_lane_mask(hls_image)
    # Edge Saturation channel and rs binary image
    lines = create_binary_image(rs_binary, edge_image[:, :, 2].astype(np.uint8))
    # Display output
    cv2.imshow('lines', lines)
    cv2.imshow('org', bgr_image)

    # End Application
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#
# Run as script
if __name__ == "__main__":
    main()
