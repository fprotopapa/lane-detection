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


#
#
# Main Application
def main():
    # Get input images
    image_files = utils.get_list_of_input_files(image_folder_name, "jpg")
    # Open test files
    # Index: 2, 5, 6
    image = cv2.imread(image_files[5])
    #
    # Start processing
    #
    # Convert to HLS (Hue, Lightness, Saturation)
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # Convert to binary
    _, binary_frame = cv2.threshold(hls_image[:, :, 1], 110, 255, cv2.THRESH_BINARY)
    edge_frame = det.get_sobel_edge_detection(binary_frame)
    # Add blur
    hls_image = cv2.GaussianBlur(hls_image, (5, 5), 0)
    # Use lightness and saturation channels for more features
    _, s_binary = cv2.threshold(hls_image[:, :, 1], 80, 255, cv2.THRESH_BINARY)
    _, r_threshold = cv2.threshold(hls_image[:, :, 2], 120, 255, cv2.THRESH_BINARY)
    rs_binary = cv2.bitwise_and(s_binary, r_threshold)
    lines = cv2.bitwise_or(rs_binary, edge_frame.astype(np.uint8))
    # Display output
    cv2.imshow('original', hls_image)
    cv2.imshow('hue', hls_image[:, :, 0])
    cv2.imshow('lightness', hls_image[:, :, 1])
    cv2.imshow('saturation', hls_image[:, :, 2])
    cv2.imshow('edges', edge_frame)
    cv2.imshow('lanes', lines)
    # End Application
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#
# Run as script
if __name__ == "__main__":
    main()
