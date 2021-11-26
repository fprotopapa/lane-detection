##########################################################
# detect_on_video.py
#
# SPDX-FileCopyrightText: Copyright 2021 Fabbio Protopapa
#
# SPDX-License-Identifier: MIT
#
# Lane detection on video files
#
# ########################################################
#
# Import libraries
import cv2

#
import numpy as np
import pickle
import os

#import lane_detection
import lane_detection.utils as utils
import lane_detection.frame_transformer as ftf

#
#
# User defined configurations
width = 1280
height = 720
is_video = False
is_calibration = True
video_folder_name = 'input_video'
video_file_format = 'mp4'
image_folder_name = 'input_image'
image_file_format = 'jpg'
calibration_data_path = os.path.join('lane_detection', 'calibration_data', "calibration.p")

def process_video(vid_files, mtx, dist, Ftf):
    # For manual application exit
    end_application = False
    # Loop through found video files
    for vid in vid_files:
        # Open next video file
        indx = 0
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print("Error opening file {}".format(vid))
        filename = utils.get_filename(vid)
        #detector = lane_detection.LaneDetection(cap)
        #detector.get_video_information(filename)
        # Enter video processing
        processed_frames = 0
        start_tick_vid = cv2.getTickCount()
        while cap.isOpened():
            # Read next frame
            ret, bgr_frame = cap.read()
            # Check for error while retrieving frame
            if ret:
                process_lane_detection(bgr_frame, mtx, dist, Ftf, filename)
                indx += 1
                if indx > 99:
                    indx = 0
                # 'ESC' to skip to next file and 'q' to end application
                pressed_key = cv2.waitKey(1) & 0xFF
                if pressed_key == 27:
                    break
                elif pressed_key == ord('q'):
                    end_application = True
                    break
            else:
                # Error while retrieving frame or video ended
                break
        # Process time for video file
        processed_frames += 1
        print(
            """     
            FPS post
            Process:  {}
            ----------------------
            """.format(int(1 / utils.get_passed_time(
                start_tick_vid, cv2.getTickCount(), processed_frames))
                        ))
        # Release video file
        cap.release()
        # Check for manual end command
        if end_application:
            break

def process_image(image_files, mtx, dist, Ftf):
    for image_path in image_files:
        filename = utils.get_filename(image_path)
        image = cv2.imread(image_path)
        process_lane_detection(image, mtx, dist, Ftf, filename)
        # End Application
        pressed_key = cv2.waitKey(0) & 0xFF
        if pressed_key == ord('q'):
            break
    cv2.destroyAllWindows()

def process_lane_detection(bgr_frame, mtx, dist, Ftf, filename):
    #
    # Lane Detection
    #
    # Distort frame
    undist_frame = Ftf.undistort_image(bgr_frame, mtx, dist)
    # Select ROI
    
    #vertices = set_polygon(undist_frame)
    vertices = [(360, 630), (568, 499), (847, 494), (1017, 623)]
    vert_poly = np.array([[vertices[0], vertices[1], vertices[2], vertices[3]]], dtype=np.int32)
    vert_trans = np.array([[vertices[1], vertices[0], vertices[2], vertices[3]]], dtype=np.float32)
    #print(vertices)
    roi_frame = Ftf.region_of_interest(undist_frame, vert_poly)
    trans_frame = Ftf.transform_image(roi_frame, vert_trans)
    # Convert color space
    # ToDO: Adjust from threshold
    bright_frame = Ftf.adjust_image(trans_frame, 1.1)
    hls_frame = Ftf.bgr_to_x(bright_frame, 'hls')
    gray_frame= Ftf.bgr_to_x(trans_frame, 'gray')
    # Edge detection
    blur_frame = Ftf.apply_gaussian_blur(gray_frame)
    canny_frame = Ftf.apply_canny_edge_det(blur_frame)
    sobel_frame = Ftf.apply_sobel_edge_det(blur_frame)
    # White color mask
    lower = np.uint8([0, 210, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = Ftf.create_mask(hls_frame, (lower, upper))
    # yellow color mask
    lower = np.uint8([10, 0, 110])
    upper = np.uint8([40, 255, 255])
    yellow_mask = Ftf.create_mask(hls_frame, (lower, upper))
    # combine the mask
    comb_mask = Ftf.combine_mask(white_mask, yellow_mask)
    intersect = Ftf.intersect_mask(comb_mask, sobel_frame)
    # Display results
    cv2.imshow("original", bgr_frame)
    cv2.setWindowTitle("original", filename)
    cv2.imshow("intersect", intersect)
    cv2.setWindowTitle("intersect", filename)
   


#
# Main application
def main():
    Ftf = ftf.FrameTransformer()
    # Print General Version information
    utils.print_version_information()
    # Search for available video files
    if not is_video:
        folder_name = image_folder_name
        file_format = image_file_format
    else:
        folder_name = video_folder_name
        file_format = video_file_format
    files = utils.get_list_of_input_files(folder_name, file_format)
    # Load calibration data
    if is_calibration:
        import pickle
        objpoints, imgpoints = pickle.load(open(calibration_data_path, "rb"))
        mtx, dist = Ftf.calibrate_camera(width, height, objpoints, imgpoints) 
    # Application start time
    start_tick_app = cv2.getTickCount()
    if is_video:
        process_video(files, mtx, dist, Ftf)
    else:
        process_image(files, mtx, dist, Ftf)
    # Total application runtime
    print("Total processing time: {}s".format(utils.get_passed_time(start_tick_app, cv2.getTickCount())))
    # Close Window
    cv2.destroyAllWindows()


#
# Run as script
if __name__ == "__main__":
    main()
