##########################################################
# lane_detector.py
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
    def __init__(self, is_video=False, width=1280, height=720, draw_area = True):
        self.vertices = None
        self.is_video = is_video
        self.width = width
        self.height = height
        self.frame = None
        self.draw_area_err = True
        self.draw_area = draw_area
        self.road_color = (204, 255, 153)
        self.lane_color = (0, 0, 255)
        self.lane_thickness = 30
        self.n_windows = 9
        self.margin = 100
        self.px_threshold = 50

    """  General methods for setting files and getting information """

    def set_next_frame(self, frame):
        self.frame = frame

    def set_vertices(self, vertices):
        self.vertices = vertices

    """ Find lanes """

    def calculate_histogram(self, frame):
        return np.sum(frame, axis=0)

    def get_hist_peaks(self, histogram):
        center = np.int(histogram.shape[0]//2)
        left_peak = np.argmax(histogram[:center])
        right_peak = np.argmax(histogram[center:]) + center
        return left_peak, right_peak

    def find_lanes(self, frame):
        histogram = self.calculate_histogram(frame)
        left_peak, right_peak = self.get_hist_peaks(histogram)
        left_fit, right_fit = self.fit_polynomial(frame, left_peak, right_peak)
        return self.draw_lanes(frame, left_fit, right_fit)


    def sliding_window(self, frame, left_peak, right_peak):
        # Set window height
        window_height = np.int(frame.shape[0]//self.n_windows)
        # Find non-zero values
        nonzero = frame.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        # Current positions to be updated later for each window in n_windows
        leftx_current = left_peak
        rightx_current = right_peak
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(self.n_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = frame.shape[0] - (window + 1) * window_height
            win_y_high = frame.shape[0] - window * window_height
            # Find the four below boundaries of the window 
            win_xleft_low = leftx_current - self.margin  
            win_xleft_high = leftx_current + self.margin  
            win_xright_low =  rightx_current - self.margin 
            win_xright_high = rightx_current + self.margin  
            # Identify the nonzero pixels in x and y within the window 
            good_left_inds = ((nonzero_y >= win_y_low ) & (nonzero_y < win_y_high) &\
                                (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low ) & (nonzero_y < win_y_high) &\
                                (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > px_threshold pixels, recenter next window 
            # (`right` or `leftx_current`) on their mean position 
            if len(good_left_inds) > self.px_threshold:
                leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > self.px_threshold:
                rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
        # Extract left and right line pixel positions
        leftx = nonzero_x[left_lane_inds]
        lefty = nonzero_y[left_lane_inds] 
        rightx = nonzero_x[right_lane_inds]
        righty = nonzero_y[right_lane_inds]
        return leftx, lefty, rightx, righty

    def fit_polynomial(self, frame, left_peak, right_peak):
        # Find our lane pixels first
        leftx, lefty, rightx, righty = self.sliding_window(frame, left_peak, right_peak)
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except:
            # Empty vector
            left_fit = [0, 0, 0]
            self.draw_area_err = False
        try:
            right_fit = np.polyfit(righty, rightx, 2)
        except:
            # Empty vector
            right_fit = [0, 0, 0]
            self.draw_area_err = False
        return left_fit, right_fit

    def draw_lanes(self, warped_frame, left_fit, right_fit):
        # Convert to 3 channels
        frame_3channel = cv2.cvtColor(warped_frame, cv2.COLOR_GRAY2BGR)
        # Generate axis for polynomial
        frame_height = np.linspace(0, frame_3channel.shape[0] - 1, frame_3channel.shape[0])
        # Frames to save results
        lanes = np.zeros_like(frame_3channel).astype(np.uint8)
        area = np.zeros_like(frame_3channel).astype(np.uint8)
        # Calculate points of lanes a*x^2 + b*x + c
        left_lane = left_fit[0] * frame_height**2 + left_fit[1] * frame_height + left_fit[2]
        right_lane = right_fit[0] * frame_height**2 + right_fit[1] * frame_height + right_fit[2]
        # Make points useable for polylines and fillpoly
        pts_left = np.array([np.transpose(np.vstack([left_lane, frame_height]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane, frame_height])))])
        # Draw the lane onto the warped blank image
        cv2.polylines(lanes, np.int_(pts_left), False, self.lane_color, self.lane_thickness)
        cv2.polylines(lanes, np.int_(pts_right), False, self.lane_color, self.lane_thickness)
        frame_3channel = cv2.addWeighted(frame_3channel, 1, lanes, 1, 0)
        if self.draw_area and self.draw_area_err:
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(area, np.int_([pts]), self.road_color)
            frame_3channel = cv2.addWeighted(frame_3channel, 1, area, 0.3, 0)
        if not self.draw_area_err:
            self.draw_area_err = True
        return frame_3channel

