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
import lane_detection.config as cfg
from lane_detection.get_coordinates import set_polygon

DEBUG = False
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
    global START_TICK_FRAME
    # For manual application exit
    end_application = False
    # Loop through found video files
    for vid in vid_files:
        # Open next video file
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print("Error opening file {}".format(vid))
        filename = utils.get_filename(vid)
        # Save output
        # capture = cv2.VideoWriter(str(filename) + '.avi', 
        #                  cv2.VideoWriter_fourcc(*'MJPG'),
        #                  20, (Detector.width, Detector.height))
        # Reset detector for next clip
        Detector.reset_detector()
        # Check video orientation
        flip = False
        if cap.get(cv2.CAP_PROP_FRAME_WIDTH) != width:
            flip = True
        # Display file information
        get_video_information(cap, filename)
        # Enter video processing
        Detector.vertices = None
        processed_frames = 0
        while cap.isOpened():
            START_TICK_FRAME = cv2.getTickCount()
            # Read next frame
            ret, bgr_frame = cap.read()
            # Flip if necessary
            if flip:
                bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)#cv2.ROTATE_90_COUNTERCLOCKWISE
            # Check for error while retrieving frame
            if ret:
                # Number of frames
                processed_frames += 1
                # Start lane detection
                result = process_lane_detection(bgr_frame, mtx, dist, Ftf, filename)
                # Write clip
                #capture.write(result)
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
        # Release video file
        cap.release()
        # Check for manual end command
        if end_application:
            break

#
# Image pipeline
def process_image(image_files, mtx, dist, Ftf):
    global START_TICK_FRAME
    for image_path in image_files:
        START_TICK_FRAME = cv2.getTickCount()
        filename = utils.get_filename(image_path)
        image = cv2.imread(image_path)
        # Reset detector for next image
        Detector.reset_detector()
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
    global START_TICK_FRAME
    # Distort frame
    if is_calibration:
        undist_frame = Ftf.undistort_frame(bgr_frame, mtx, dist)
    else:
        undist_frame = bgr_frame
    # Select ROI
    if is_man_roi or Detector.vertices is None:
        # Check for saved ROI
        if not is_man_roi:
            try:
                Detector.vertices = utils.save_load_np_var(filename, save = False)
            except:
                Detector.vertices = None
        if is_man_roi and Detector.vertices is None:
            vertices = set_polygon(undist_frame)
            Detector.set_vertices(vertices)
        elif is_video and Detector.vertices is None:
            vertices = set_polygon(undist_frame)
            Detector.set_vertices(vertices)
            utils.save_load_np_var(filename, data = vertices)
        elif not is_video and Detector.vertices is None:
            vertices = set_polygon(undist_frame)
            Detector.set_vertices(vertices)
            utils.save_load_np_var(filename, data = vertices)
        else:
            vertices = Detector.vertices
    else:
        vertices = Detector.vertices
    vert_poly = np.array([[vertices[0], vertices[1], vertices[2], vertices[3]]], dtype=np.int32)
    vert_trans = np.array([[vertices[1], vertices[0], vertices[2], vertices[3]]], dtype=np.float32)
    # Generate ROI on frame and tranform to Bird-Eye view
    roi_frame = Ftf.region_of_interest(undist_frame, vert_poly)
    trans_frame, M, minv = Ftf.transform_frame(roi_frame, vert_trans)
    # Adjust brightness
    bright_fac = Ftf.brightness_estimation(trans_frame)
    bright_frame = Ftf.adjust_frame(trans_frame, bright_fac)
    # Convert color space
    hls_frame = Ftf.bgr_to_x(bright_frame, 'hls')
    gray_frame= Ftf.bgr_to_x(trans_frame, 'gray')
    # Edge detection
    blur_frame = Ftf.apply_gaussian_blur(gray_frame)
    #canny_frame = Ftf.apply_canny_edge_det(blur_frame)
    sobel_frame = Ftf.apply_sobel_edge_det(blur_frame)
    filter_frame = sobel_frame
    # White color mask
    lower = np.uint8(cfg.white_lower) # [200, 200, 200])
    upper = np.uint8(cfg.white_upper) # [255, 255, 255])
    white_mask = Ftf.create_mask(bright_frame, (lower, upper))
    # Yellow color mask
    lower = np.uint8(cfg.yellow_lower)
    upper = np.uint8(cfg.yellow_upper)
    yellow_mask = Ftf.create_mask(hls_frame, (lower, upper))
    # Combine the mask
    comb_mask = Ftf.combine_frames(white_mask, yellow_mask)
    intersect = Ftf.intersect_mask(comb_mask, filter_frame)
    # ToDo average, curve prediction       
    # Draw lanes
    lanes, avg_rad = Detector.find_lanes(intersect)

    if avg_rad is not None:
        Detector.insert_direction(bgr_frame, avg_rad)
    # Return from Bird-eye view to normal view
    unwarp = Ftf.untransform_frame(lanes, minv)
    # Overlay drawings on bgr frame
    result = cv2.addWeighted(bgr_frame, 1, unwarp, 1, 0)
    # FPS
    fps = 1 / (utils.get_passed_time(
                            START_TICK_FRAME, 
                            cv2.getTickCount()))
    Detector.insert_fps(result, fps)

    # Display results
    if DEBUG:
        cv2.imshow("Combined", comb_mask)
        cv2.imshow("White", white_mask)
        cv2.imshow("Edge", filter_frame)
        cv2.imshow("Bright", bright_frame)
        cv2.imshow("original", bgr_frame)
        cv2.imshow("intersect", intersect)
        cv2.imshow("lanes", unwarp)
        cv2.imshow("result", result)
        cv2.setWindowTitle("result", filename)
    else:
        cv2.imshow("result", result)
        cv2.setWindowTitle("result", filename)
    return result


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
    if is_video:
        process_video(files, mtx, dist, Ftf)
    else:
        process_image(files, mtx, dist, Ftf)
    # Close Window
    cv2.destroyAllWindows()


#
# Run as script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Start lane detection. Press <q> to quit and <ESC> for next frame.')
    parser.add_argument('--image', type=bool, default=False,
                    help='Detect lanes on video files or images <False>')
    parser.add_argument('--nocal', type=bool, default=False,
                    help='Don\' calibrate input files <False>')
    parser.add_argument('--format', type=str, default=None,
                    help='File format <"mov">/<"jpg">')
    parser.add_argument('--size', nargs=2, type=int, default=[1280, 720],
                    help='Image width and height <1280 720>')
    parser.add_argument('--roi', type=bool, default=False,
                    help='Manual ROI selection <False>')
    args = parser.parse_args()
    
    is_video = not args.image
    is_calibration = not args.nocal
    is_man_roi = args.roi
    width = args.size[0]
    height = args.size[1]
    
    START_TICK_FRAME = 0

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
    # Object for finding and drawing lanes
    Detector = det.LaneDetector(is_video=is_video, queue_len=cfg.queue_len)
    # Drawing
    Detector.draw_area = cfg.draw_area
    Detector.l_lane_color = cfg.l_lane_color
    Detector.r_lane_color = cfg.r_lane_color
    Detector.lane_thickness = cfg.lane_thickness
    Detector.road_color = cfg.road_color
    # Frame dimensions
    Detector.width = cfg.width
    Detector.height = cfg.height
    # Sliding window
    Detector.n_windows = cfg.n_windows
    Detector.margin = cfg.margin
    Detector.nb_margin = cfg.nb_margin
    Detector.nb_margin = cfg.nb_margin
    Detector.radii_threshold = cfg.radii_threshold
    # Conversion pixel to meter
    Detector.px_to_m_y = cfg.px_to_m_y
    Detector.px_to_m_x = cfg.px_to_m_x
    # Lanes and poly
    Detector.min_lane_dis = cfg.min_lane_dis
    Detector.poly_thr_a = cfg.poly_thr_a
    Detector.poly_thr_b = cfg.poly_thr_b
    Detector.poly_thr_c = cfg.poly_thr_c
    # Start process
    main()
