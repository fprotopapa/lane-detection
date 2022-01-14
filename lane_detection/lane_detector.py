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
import math
from collections import deque 

class LaneDetector:
    def __init__(self, is_video=False, width=1280, height=720, draw_area = True, queue_len=10):
        # Roi
        self.vertices = None
        # Video pipline
        self.is_video = is_video
        # Frame dimension
        self.width = width
        self.height = height
        # Draw 
        self.draw_area_err = True
        self.draw_area = draw_area
        self.road_color = (204, 255, 153)
        self.l_lane_color = (0, 0, 255)
        self.r_lane_color = (255, 0, 0)
        self.lane_thickness = 30
        # Lane search
        self.n_windows = 9
        self.margin = 100
        self.nb_margin = 100
        self.px_threshold = 50
        self.radii_threshold = 10
        self.min_lane_dis = 600
        # Current lanes and radii
        self.l_curr_fit = None
        self.r_curr_fit = None
        self.l_diff_fit = 0
        self.r_diff_fit = 0
        self.l_curr_cr = 0
        self.r_curr_cr = 0
        self.lost_track = 0
        self.lost_radii = 0
        self.poly_thr_a = 0.001
        self.poly_thr_b = 0.4
        self.poly_thr_c = 150
        # Convert px to meter
        self.px_to_m_y = 30/720 # meters per pixel in y dimension
        self.px_to_m_x = 3.7/700 # meters per pixel in x dimension
        # Averaging
        self.queue_len = queue_len
        self.l_fit_que = deque(maxlen=self.queue_len)
        self.r_fit_que = deque(maxlen=self.queue_len)
        self.l_rad_que = deque(maxlen=self.queue_len)
        self.r_rad_que = deque(maxlen=self.queue_len)
        self.weights = np.arange(1, self.queue_len + 1) / self.queue_len
        # No Text on frame
        self.no_text = False


    """  General methods for setting files and getting information """
    def set_vertices(self, vertices):
        self.vertices = vertices

    def reset_detector(self):
        self.empty_queue()
        self.vertices = None
        self.l_curr_fit = None
        self.r_curr_fit = None
        self.l_diff_fit = 0
        self.r_diff_fit = 0
        self.l_curr_cr = 0
        self.r_curr_cr = 0
        self.lost_track = 0
        self.lost_radii = 0

    def empty_queue(self):
        self.l_fit_que = deque(maxlen=self.queue_len)
        self.r_fit_que = deque(maxlen=self.queue_len)
        self.l_rad_que = deque(maxlen=self.queue_len)
        self.r_rad_que = deque(maxlen=self.queue_len)
    """ Find lanes """

    def calculate_histogram(self, frame):
        return np.sum(frame, axis=0)

    def get_hist_peaks(self, histogram):
        center = np.int(histogram.shape[0]//2)
        left_peak = np.argmax(histogram[:center])
        right_peak = np.argmax(histogram[center:]) + center
        return left_peak, right_peak

    def cr_to_degree(self, cr, arc_length):
        dc = (180 * arc_length) / (math.pi * cr)
        return dc/2

    def find_lanes(self, frame):
        self.check_track()
        if self.l_curr_fit is None or self.r_curr_fit is None:
            self.empty_queue()
            histogram = self.calculate_histogram(frame)
            left_peak, right_peak = self.get_hist_peaks(histogram)
            leftx, lefty, rightx, righty = self.sliding_window(frame, left_peak, right_peak)
            left_fit, right_fit = self.fit_polynomial(leftx, lefty, rightx, righty)
            left_fit_cr, right_fit_cr = self.fit_polynomial(
                                        leftx * self.px_to_m_x, lefty * self.px_to_m_y, 
                                        rightx * self.px_to_m_x, righty * self.px_to_m_y)
            # Get radii of lane curvature
            left_rad, right_rad = self.calculate_poly_radii(frame, left_fit_cr, right_fit_cr)
            self.r_curr_cr = left_rad
            self.l_curr_cr = right_rad
            self.r_curr_fit = right_fit
            self.l_curr_fit = left_fit
            self.l_fit_que.append(left_fit)
            self.r_fit_que.append(right_fit)
            self.l_rad_que.append(left_rad)
            self.r_rad_que.append(right_rad)
        else:
            left_fit, right_fit, left_fit_cr, right_fit_cr, _ = self.nearby_search(
                                                                    frame, 
                                                                    np.average(self.l_fit_que, 0, self.weights[-len(self.l_fit_que):]), 
                                                                    np.average(self.r_fit_que, 0, self.weights[-len(self.r_fit_que):]))
            self.l_fit_que.append(left_fit)
            self.r_fit_que.append(right_fit)
        avg_rad = round(np.mean([np.average(self.r_rad_que, 0, self.weights[-len(self.r_rad_que):]), 
                                np.average(self.l_rad_que, 0, self.weights[-len(self.l_rad_que):])]),0)
        
        try:
            return (self.draw_lanes(frame, 
                                    np.average(self.l_fit_que, 0, self.weights[-len(self.l_fit_que):]), 
                                    np.average(self.r_fit_que, 0, self.weights[-len(self.r_fit_que):])),
                    avg_rad)
        except:
            return (np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)), None)


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

    def calculate_poly_radii(self, frame, left_fit, right_fit):
        frame_height = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
        max_px_window = np.max(frame_height)
        try:
            left_rad = ((1 + (2 * left_fit[0] * max_px_window * self.px_to_m_y + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
            right_rad = ((1 + (2 * right_fit[0] * max_px_window * self.px_to_m_y + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])
            if math.isinf(left_rad) or math.isinf(right_rad):
                return self.l_curr_cr, self.r_curr_cr
        except:
            return self.l_curr_cr, self.r_curr_cr
        return int(left_rad), int(right_rad)

    def check_radii(self, left_rad, right_rad):
        avg_l = np.average(self.l_rad_que, 0, self.weights[-len(self.l_rad_que):])
        avg_r = np.average(self.r_rad_que, 0, self.weights[-len(self.r_rad_que):])
        abs_l__diff = np.absolute(avg_l - left_rad)
        abs_r__diff = np.absolute(avg_r - right_rad)
        if abs_l__diff > (avg_l / self.radii_threshold) and self.lost_radii < 5 and abs_r__diff > (avg_r / self.radii_threshold):
            self.lost_radii += 1
            return False
        else:
            self.lost_radii = 0
            return True

    def fit_polynomial(self, leftx, lefty, rightx, righty):
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except:
            # Empty vector
            left_fit = self.l_curr_fit
            self.draw_area_err = False
        try:
            right_fit = np.polyfit(righty, rightx, 2)
        except:
            # Empty vector
            right_fit = self.r_curr_fit
            self.draw_area_err = False
        return left_fit, right_fit

    def insert_direction(self, frame, avg_rad):
        if not self.no_text:
            cv2.putText(frame, 'Curvature radius: {:.2f} m'.format(avg_rad), 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        else:
            self.no_text = False

    def insert_fps(self, frame, fps):
        cv2.putText(frame, 'FPS: {}'.format(int(fps)), 
            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    def check_track(self):
        if self.lost_track > 5:
            print('Reset tracks')
            self.l_curr_fit = None
            self.r_curr_fit = None
            self.lost_track = 0
            self.no_text = True

    def draw_lanes(self, warped_frame, left_fit, right_fit):
        # Convert to 3 channels
        frame_3channel = cv2.cvtColor(np.zeros_like(warped_frame), cv2.COLOR_GRAY2BGR)
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
        #print(pts_right[0][len(pts_right[0]) - 20][0] - pts_left[0][len(pts_left[0]) - 20][0])
        if (pts_right[0][len(pts_right[0]) - 20][0] - pts_left[0][len(pts_left[0]) - 20][0]) < self.min_lane_dis:
            return frame_3channel
        # Draw the lane onto the warped blank image
        cv2.polylines(lanes, np.int_(pts_left), False, self.l_lane_color, self.lane_thickness)
        cv2.polylines(lanes, np.int_(pts_right), False, self.r_lane_color, self.lane_thickness)
        frame_3channel = cv2.addWeighted(frame_3channel, 1, lanes, 1, 0)
        
        if self.draw_area and self.draw_area_err:
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(area, np.int_([pts]), self.road_color)
            frame_3channel = cv2.addWeighted(frame_3channel, 1, area, 0.3, 0)
        if not self.draw_area_err:
            self.draw_area_err = True
        return frame_3channel

    def nearby_search(self, frame, left_fit, right_fit):
        
        nonzero = frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.nb_margin

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
            & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
            & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        try:
            left_fit, right_fit = self.fit_polynomial(leftx, lefty, rightx, righty)
        except:
            print("Couldn't fit polynomial.")
            self.lost_track += 1
            return self.l_curr_fit, self.r_curr_fit, self.l_curr_cr, self.r_curr_cr, None
        else:
            # Check difference in fit coefficients between last and new fits  
            try:
                self.l_diff_fit = self.l_curr_fit - left_fit
                self.r_diff_fit = self.r_curr_fit - right_fit
            except:
                self.l_diff_fit = 0
                self.r_diff_fit = 0
            if (self.l_diff_fit[0]>self.poly_thr_a or self.l_diff_fit[1]>self.poly_thr_b or self.l_diff_fit[2]>self.poly_thr_c):
                self.lost_track += 1
                print("Left lane threshold exceeded.")
                self.no_text = True
                return self.l_curr_fit, self.r_curr_fit, self.l_curr_cr, self.r_curr_cr, None
            if (self.r_diff_fit[0]>self.poly_thr_a or self.r_diff_fit[1]>self.poly_thr_b or self.r_diff_fit[2]>self.poly_thr_c):
                self.lost_track += 1
                print("Right lane threshold exceeded.")
                self.no_text = True
                return self.l_curr_fit, self.r_curr_fit, self.l_curr_cr, self.r_curr_cr, None
            # Reset counter
            self.lost_track = 0
            # Update current fit
            self.l_curr_fit = left_fit
            self.r_curr_fit = right_fit
            # Fit new polynomials to x,y in world space
            try:
                left_fit_cr, right_fit_cr = self.fit_polynomial(
                                leftx * self.px_to_m_x, lefty * self.px_to_m_y, 
                                rightx * self.px_to_m_x, righty * self.px_to_m_y)
            except:
                return self.l_curr_fit, self.r_curr_fit, self.l_curr_cr, self.r_curr_cr, None       
            left_cr, right_cr = self.calculate_poly_radii(frame, left_fit_cr, right_fit_cr)

            if self.check_radii(left_cr, right_cr):
                self.l_curr_cr = left_cr
                self.r_curr_cr = right_cr
                self.l_rad_que.append(left_cr)
                self.r_rad_que.append(right_cr)
            else: 
                return left_fit, right_fit, self.l_curr_cr, self.r_curr_cr, None
            return left_fit, right_fit, left_cr, right_cr, None

def main():
    print("No main.")


#
# Run as script
if __name__ == "__main__":
    main()