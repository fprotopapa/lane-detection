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

import lane_detection
import utils

#
# User defined configurations
image_folder_name = 'input_image'


#
# Main Application
def main():
    # Get input images
    image_files = utils.get_list_of_input_files(image_folder_name, "jpg")
    # Open test files
    # Index: 2, 5, 6
    bgr_image = cv2.imread(image_files[0])
    filename = utils.get_filename(image_files[0])


    # Distortion and warp
    # Four corners of the book in source image
    #                   TL              BL          BR          TR
    #pts_src = np.array([[440, 600], [680, 235], [1062, 670], [440, 675]])
    # Four corners of the book in destination image.
    pts_dst = np.array([[100, 100],[600, 100],[600, 400],[100, 400]])
    '''
    pts_src and pts_dst are numpy arrays of points
    in source and destination images. We need at least
    4 corresponding points.
    https://learnopencv.com/homography-examples-using-opencv-python-c/
    https://dsp.stackexchange.com/questions/19907/is-it-possible-to-hack-camera-calibration-without-having-access-to-the-camera
    '''
    #h, status = cv2.findHomography(pts_src, pts_dst)

    '''
    The calculated homography can be used to warp
    the source image to destination. Size is the
    size (width,height) of im_dst
    '''
    #bgr_image = cv2.warpPerspective(bgr_image, h, (bgr_image.shape[1],bgr_image.shape[0]))

    #cv2.imshow('warp', bgr_image)
    detector = lane_detection.LaneDetection(bgr_image, False)
    detector.get_image_information(filename)
    #
    # Start processing
    #
    # Convert to HLS (Hue, Lightness, Saturation)
    hls_image = detector.bgr_to_hls(bgr_image)

    mask_lanes = detector.create_hls_lane_mask(hls_image)

    masked_frame = detector.apply_mask(bgr_image, mask_lanes)
    gray_frame = detector.bgr_to_gray(masked_frame)

    # Reduce image to ROI
    # Set ROIs
    #detector.set_bottom_roi(hls_image)
    # Mask top of image
    # offset_street = detector.set_top_roi(hls_image, bgr_image, width, height)
    # hls_image[:int(height * offset_street), :, :] = (0, 0, 0)
    # Edge detection
    edge_image = detector.apply_gaussian_blur(gray_frame, (15, 15))
    edge_image = detector.apply_canny_edge_det(edge_image)
    # Use lightness and saturation channels for detecting lanes
    #rs_binary = detector.create_binary_lane_mask(hls_image)
    # Edge Saturation channel and rs binary image
    #lines = detector.create_binary_image(rs_binary, edge_image[:, :, 2].astype(np.uint8))
    # Display output
    cv2.imshow('mask', mask_lanes)
    cv2.imshow('masked', masked_frame)
    cv2.imshow('lines', edge_image)
    cv2.imshow('org', bgr_image)

    # End Application
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#
# Run as script
if __name__ == "__main__":
    main()
