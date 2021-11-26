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
    def apply_sobel_edge_det(frame):
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
    def create_mask(image, threshold):
        return cv2.inRange(image, threshold[0], threshold[1])

    @staticmethod
    def apply_threshold(channel, threshold=(125, 255), thresh_typ=cv2.THRESH_BINARY):
        _, binary = cv2.threshold(channel, threshold[0], threshold[1], thresh_typ)
        return binary

    @staticmethod
    def combine_mask(channel, mask):
        return cv2.bitwise_or(channel, channel, mask=mask)

    @staticmethod
    def intersect_mask(channel, mask):
        return cv2.bitwise_and(channel, channel, mask=mask)

    @staticmethod
    def adjust_image(image, brightness=1.0, contrast=0):
        # brightness 0 - 100
        beta = brightness * 100
        # contrast 1.0 - 3.0
        alpha = (contrast * 2) + 1
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def region_of_interest(image, vertices):
        #defining a blank mask to start with
        mask = np.zeros_like(image)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        return cv2.bitwise_and(image, mask)
    
    @staticmethod
    def calibrate_camera(w, h, objpoints, imgpoints):
        _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
        return mtx, dist

    @staticmethod
    def undistort_image(image, mtx, dist):
        return cv2.undistort(image, mtx, dist, None, mtx)

    @staticmethod
    def transform_image(image, vertices, offset=0):
        img_size = (image.shape[1], image.shape[0])
        src = vertices
        warped_leftupper = (offset,0)
        warped_rightupper = (offset, image.shape[0])
        warped_leftlower = (image.shape[1] - offset, 0)
        warped_rightlower = (image.shape[1] - offset, image.shape[0])
        dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])

        # calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        minv = cv2.getPerspectiveTransform(dst, src)
        
        # Warp the image
        warped = cv2.warpPerspective(image, M, img_size, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
        return np.copy(warped)


def main():
    import utils
    from get_coordinates import set_polygon
    import matplotlib.pyplot as plt
    import pickle
    import os

    objpoints, imgpoints = pickle.load( open(os.path.join('lane_detection', 'calibration_data', "calibration.p"), "rb" ) )
    image_folder_name = 'test_image'
    image_paths = utils.get_list_of_input_files(image_folder_name, "jpg")
    images = [cv2.imread(image_path) for image_path in image_paths]
    image = images[0]
    Ftf = FrameTransformer()
    

    h, w, _ = image.shape
    mtx, dist = Ftf.calibrate_camera(w, h, objpoints, imgpoints)
    undist_image = Ftf.undistort_image(image, mtx, dist)
    # warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower]
    # left_bottom, left_top, right_top, right_bottom
    #vertices = set_polygon(undist_image)
    vertices = [(360, 630), (568, 499), (847, 494), (1017, 623)]
    vert_poly = np.array([[vertices[0], vertices[1], vertices[2], vertices[3]]], dtype=np.int32)
    #vert_trans = np.array([[vertices[1], vertices[2], vertices[0], vertices[3]]], dtype=np.float32)
    vert_trans = np.array([[vertices[1], vertices[0], vertices[2], vertices[3]]], dtype=np.float32)
    #print(vertices)
    roi_image = Ftf.region_of_interest(undist_image, vert_poly)
    trans_image = Ftf.transform_image(image, vert_trans)

    bright_image = Ftf.adjust_image(trans_image, 1.1)
    hls_image = Ftf.bgr_to_x(bright_image, 'hls')
    gray_image= Ftf.bgr_to_x(trans_image, 'gray')

    blur_image = Ftf.apply_gaussian_blur(gray_image)
    canny_image = Ftf.apply_canny_edge_det(blur_image)
    sobel_image = Ftf.apply_sobel_edge_det(blur_image)
    
    # White color mask
    lower = np.uint8([0, 210, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = Ftf.create_mask(hls_image, (lower, upper))
    # yellow color mask
    lower = np.uint8([10, 0, 110])
    upper = np.uint8([40, 255, 255])
    yellow_mask = Ftf.create_mask(hls_image, (lower, upper))
    # combine the mask
    comb_mask = Ftf.combine_mask(white_mask, yellow_mask)

    intersect = Ftf.intersect_mask(comb_mask, sobel_image)

    utils.imshow_images(image, 'Org')  
    # imshow_images(undist_image, 'Calib')
    # imshow_images(roi_image, 'ROI')
    # # imshow_images(T.adjust_image(image, 1), 'Adj')
    utils.imshow_images(trans_image, 'trans')
    utils.imshow_images(hls_image, 'HLS')
    utils.imshow_images(blur_image, 'Gaus Blur')
    utils.imshow_images(canny_image, 'Canny')
    # # imshow_images(T.apply_threshold(image[:,:,1]), 'Binary')
    utils.imshow_images(sobel_image, 'Sobel')
    utils.imshow_images(comb_mask, 'Mask')
    utils.imshow_images(intersect, 'Intersect')
    # imshow_images(T.create_channel_mask(image[:,:,1], (125, 255)), 'Mask')





#
# Run as script
if __name__ == "__main__":
    main()