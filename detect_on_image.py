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




#
#
# Main Application
def main():
    # Get input images
    image_files = utils.get_list_of_input_files(image_folder_name, "jpg")
    # Open test files
    # Index: 2, 5, 6
    bgr_image = cv2.imread(image_files[2])
    filename = utils.get_filename(image_files[2])
    width, height, _ = det.get_image_information(bgr_image, filename)
    #
    # Start processing
    #
    # Convert to HLS (Hue, Lightness, Saturation)
    hls_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS)
    # Reduce image to ROI
    det.set_bottom_roi(hls_image, height)
    # Mask top of image
    offset_street = det.set_top_roi(hls_image, bgr_image, width, height)
    hls_image[:int(height * offset_street), :, :] = (0, 0, 0)
    # Mask lanes
    mask = create_hls_lane_mask(hls_image)
    masked_image = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
    # Convert to gray
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    # Edge detection
    gray_image = cv2.GaussianBlur(gray_image, (15, 15), 0)
    edge_image = cv2.Canny(gray_image, 50, 150)
    # Use lightness and saturation channels for more features
    _, s_binary = cv2.threshold(hls_image[:, :, 1], 80, 255, cv2.THRESH_BINARY)
    _, r_threshold = cv2.threshold(hls_image[:, :, 2], 120, 255, cv2.THRESH_BINARY)
    rs_binary = cv2.bitwise_and(s_binary, r_threshold)
    # lines = cv2.bitwise_or(rs_binary, edge_frame.astype(np.uint8))
    # Display output
    cv2.imshow('original', hls_image)
    # cv2.imshow('hue', hls_image[:, :, 0])
    # cv2.imshow('lightness', hls_image[:, :, 1])
    # cv2.imshow('saturation', hls_image[:, :, 2])
    # cv2.imshow('sobel', sobel_y)
    cv2.imshow('edges', edge_image)
    cv2.imshow('org', bgr_image)
    cv2.imshow('gray', gray_image)
    cv2.imshow('mask', masked_image)
    # End Application
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#
# Run as script
if __name__ == "__main__":
    main()
