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
    def create_mask(frame, threshold):
        return cv2.inRange(frame, threshold[0], threshold[1])

    @staticmethod
    def apply_threshold(channel, threshold=(125, 255), thresh_typ=cv2.THRESH_BINARY):
        _, binary = cv2.threshold(channel, threshold[0], threshold[1], thresh_typ)
        return binary

    @staticmethod
    def combine_mask(channel, mask):
        return cv2.bitwise_or(channel, channel, mask=mask)

    @staticmethod
    def combine_frames(frame_1, frame_2):
        return cv2.bitwise_or(frame_1, frame_2)

    @staticmethod
    def intersect_mask(channel, mask):
        return cv2.bitwise_and(channel, channel, mask=mask)

    @staticmethod
    def brightness_estimation(frame):
        mean = np.mean(frame)
        #print(mean)
        if mean < 40:
            return 1.4
        elif mean < 60:
            return 1.3
        elif mean < 80:
            return 0.3
        else:
            return 0.2
        
    @staticmethod
    def adjust_frame(frame, brightness=1.0, contrast=0):
        # brightness 0 - 100
        beta = brightness * 100
        # contrast 1.0 - 3.0
        alpha = (contrast * 2) + 1
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    @staticmethod
    def region_of_interest(frame, vertices):
        #defining a blank mask to start with
        mask = np.zeros_like(frame)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input frame
        if len(frame.shape) > 2:
            channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your frame
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the frame only where mask pixels are nonzero
        return cv2.bitwise_and(frame, mask)
    
    @staticmethod
    def calibrate_camera(w, h, objpoints, imgpoints):
        _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
        return mtx, dist

    @staticmethod
    def undistort_frame(frame, mtx, dist):
        return cv2.undistort(frame, mtx, dist, None, mtx)

    @staticmethod
    def transform_frame(frame, vertices, offset=0):
        img_size = (frame.shape[1], frame.shape[0])
        src = vertices
        warped_leftupper = (offset,0)
        warped_rightupper = (offset, frame.shape[0])
        warped_leftlower = (frame.shape[1] - offset, 0)
        warped_rightlower = (frame.shape[1] - offset, frame.shape[0])
        dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])

        # calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        minv = cv2.getPerspectiveTransform(dst, src)
        
        # Warp the frame
        warped = cv2.warpPerspective(frame, M, img_size, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
        return np.copy(warped), M, minv

    @staticmethod
    def untransform_frame(frame, minv):      
        img_size = (frame.shape[1], frame.shape[0]) 
        # Unwarp the frame
        unwarped = cv2.warpPerspective(frame, minv, img_size, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
        return np.copy(unwarped)

def main():
    print("No main.")


#
# Run as script
if __name__ == "__main__":
    main()