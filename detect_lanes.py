##########################################################
# detect_lanes.py
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
import os

#import lane_detection
import lane_detection.utils as utils
import lane_detection.frame_transformer as ftf
import lane_detection.lane_detector as det
from lane_detection.get_coordinates import set_polygon

#
# Print Video Information, return video resolution (width, height)
def get_video_information(cap, filename=None):
    if filename is None:
        filename = "-"
    print("""
    File:     {}
    FPS:      {}
    # Frames: {}
    Width:    {}
    Height:   {}
    """.format(
        filename,
        int(cap.get(cv2.CAP_PROP_FPS)),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width,
        height
    ))

#
# Print Image Information, return image resolution (width, height, channel)
def get_image_information(filename=None):
    if filename is None:
        filename = "-"
    print("""
    File:     {}
    Width:    {}
    Height:   {}
    """.format(
        filename,
        width,
        height
    ))

#
# Video pipeline
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
        # Reset selected vertices
        detector.set_vertices(None)
        # Check video orientation
        if cap.get(cv2.CAP_PROP_FRAME_WIDTH) != width:
            flip = True
        # Display file information
        get_video_information(cap, filename)
        # Enter video processing
        processed_frames = 0
        start_tick_vid = cv2.getTickCount()
        while cap.isOpened():
            # Read next frame
            ret, bgr_frame = cap.read()
            # Flip if necessary
            if flip:
                bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Check for error while retrieving frame
            if ret:
                # Number of frames
                processed_frames += 1
                # Start lane detection
                process_lane_detection(bgr_frame, mtx, dist, Ftf, filename)
                indx += 1
                if indx > 99:
                    indx = 0
                # 'ESC' to skip to next file and 'q' to end application
                pressed_key = cv2.waitKey(1) & 0xFF
                if pressed_key == 27:
                    cv2.destroyAllWindows()
                    break
                elif pressed_key == ord('q'):
                    end_application = True
                    break
            else:
                # Error while retrieving frame or video ended
                break
        # Process time for video file    
        print(
            """     
            FPS post
            Process:  {}
            ----------------------
            """.format(int(1 / (utils.get_passed_time(
                start_tick_vid, cv2.getTickCount(), processed_frames))
                        )))
        # Release video file
        cap.release()
        # Check for manual end command
        if end_application:
            break

#
# Image pipeline
def process_image(image_files, mtx, dist, Ftf):
    for image_path in image_files:
        filename = utils.get_filename(image_path)
        image = cv2.imread(image_path)
        process_lane_detection(image, mtx, dist, Ftf, filename)
        # 'ESC' to skip to next file and 'q' to end application
        pressed_key = cv2.waitKey(0) & 0xFF
        if pressed_key == 27:
            cv2.destroyAllWindows()
        elif pressed_key == ord('q'):
            break
    cv2.destroyAllWindows()

#
# Lane detection
def process_lane_detection(bgr_frame, mtx, dist, Ftf, filename):
    #
    # Lane Detection
    #
    # Distort frame
    undist_frame = Ftf.undistort_image(bgr_frame, mtx, dist)
    # Select ROI
    if is_man_roi:
        if is_video and detector.vertices is None:
            vertices = set_polygon(undist_frame)
            detector.set_vertices(vertices)
        elif is_video and detector.vertices is not None:
            vertices = detector.vertices
        else:
            vertices = set_polygon(undist_frame)
    else:
        vertices = [(360, 630), (568, 499), (847, 494), (1017, 623)]
    vert_poly = np.array([[vertices[0], vertices[1], vertices[2], vertices[3]]], dtype=np.int32)
    vert_trans = np.array([[vertices[1], vertices[0], vertices[2], vertices[3]]], dtype=np.float32)
    #print(vertices)
    # Generate ROI on frame and tranform to Bird-Eye view
    roi_frame = Ftf.region_of_interest(undist_frame, vert_poly)
    trans_frame = Ftf.transform_image(roi_frame, vert_trans)
    # ToDO: Adjust from threshold
    bright_frame = Ftf.adjust_image(trans_frame, 1.1)
    # Convert color space
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
    # Yellow color mask
    lower = np.uint8([10, 0, 110])
    upper = np.uint8([40, 255, 255])
    yellow_mask = Ftf.create_mask(hls_frame, (lower, upper))
    # Combine the mask
    comb_mask = Ftf.combine_mask(white_mask, yellow_mask)
    intersect = Ftf.intersect_mask(comb_mask, sobel_frame)
    # Display results
    cv2.imshow("original", bgr_frame)
    cv2.setWindowTitle("original", filename)
    cv2.imshow("intersect", intersect)
    cv2.setWindowTitle("intersect", filename)
    # ToDo
    # histogram, search peaks, sliding window, threshold, curve fitting, draw, output


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
    import argparse

    parser = argparse.ArgumentParser(description='Start lane detection. Press <q> to quit and <ESC> for next frame.')
    parser.add_argument('--video', type=bool, default=False,
                    help='Detect lanes on video files or images <False>')
    parser.add_argument('--cal', type=bool, default=True,
                    help='Calibrate input files <True>')
    parser.add_argument('--format', type=str, default=None,
                    help='File format <"mov">/<"jpg">')
    parser.add_argument('--size', nargs=2, type=int, default=[1280, 720],
                    help='Image width and height <1280 720>')
    parser.add_argument('--roi', type=bool, default=True,
                    help='Manual ROI selection <True>')
    args = parser.parse_args()
    
    is_video = args.video
    is_calibration = args.cal
    is_man_roi = args.roi

    width = args.size[0]
    height = args.size[1]
    
    video_file_format = 'mov'
    image_file_format = 'jpg'

    if is_video:
        video_folder_name = 'input_video'
        if args.format is not None:
            video_file_format = args.format
    else:
        image_folder_name = 'input_image'
        if args.format is not None:
            image_file_format = args.format

    calibration_data_path = os.path.join('lane_detection', 'calibration_data', "calibration.p")
    detector = det.LaneDetector(is_video=is_video)

    main()
