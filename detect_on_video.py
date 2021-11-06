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

import lane_detection
import utils

#
#
# User defined configurations
video_folder_name = "input_video"
video_file_format = "*.mp4"


#
# Main application
def main():
    # Print General Version information
    utils.print_version_information()
    # Search for available video files
    vid_files = utils.get_list_of_input_files(video_folder_name)
    # For manual application exit
    end_application = False
    # Application start time
    start_tick_app = cv2.getTickCount()
    top_roi_arr = np.zeros((100,), dtype=float)
    roi_indx = 0
    # Loop through found video files
    for vid in vid_files:
        # Open next video file
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print("Error opening file {}".format(vid))
        filename = utils.get_filename(vid)
        detector = lane_detection.LaneDetection(cap)
        detector.get_video_information(filename)
        # Enter video processing
        processed_frames = 0
        start_tick_vid = cv2.getTickCount()
        while cap.isOpened():
            # Read next frame
            ret, bgr_frame = cap.read()
            # Check for error while retrieving frame
            if ret:
                #
                # Lane Detection
                #
                detector.set_next_frame(bgr_frame)
                # Convert to HLS
                hls_frame = detector.bgr_to_hls(bgr_frame)
                #print(np.mean(hls_frame[:, :, 1]))
                #hls_frame[:,:,1] = np.where(hls_frame[:, :, 1] < 125, 120, 0)
                # Reduce image
                # Mask 10% of bottom
                detector.set_bottom_roi(hls_frame)
                #bgr_frame[int(height * 0.9):, :, :] = (0, 0, 0)
                # Mask top
                #offset = detector.set_top_roi(hls_frame, bgr_frame)
                #top_roi_arr[roi_indx] = offset
                roi_indx += 1
                if roi_indx > 99:
                    roi_indx = 0
                #hls_frame[:int(detector.height * np.mean(top_roi_arr)), :, :] = (0, 0, 0)
                # Edge detection
                edge_frame = detector.apply_gaussian_blur(hls_frame, (15, 15))
                edge_frame = detector.apply_sobel_edge_detection(edge_frame)
                # Use lightness and saturation channels for detecting lanes
                rs_binary = detector.create_binary_lane_mask(hls_frame)
                # Edge Saturation channel and rs binary image
                lines = detector.create_binary_image(rs_binary, edge_frame[:, :, 2].astype(np.uint8))

                # Mask image
                # ToDo calculate mean of street in front of car
                # calculate mean of sky -> start mask when hitting the road (add max of 0.5 image and apply threshold)
                # edge_frame[0:int(height/3), :] = (0, 0, 0)

                # Display results
                processed_frames += 1
                cv2.imshow("original_id", bgr_frame)
                cv2.setWindowTitle("original_id", filename)
                cv2.imshow("lanes_id", lines)
                cv2.setWindowTitle("lanes_id", filename)
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
    # Total application runtime
    print("Total processing time: {}s".format(utils.get_passed_time(start_tick_app, cv2.getTickCount())))
    # Close Window
    cv2.destroyAllWindows()


#
# Run as script
if __name__ == "__main__":
    main()
