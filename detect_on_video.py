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
import lane_detection as det
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
    start_time_app = cv2.getTickCount()
    # Loop through found video files
    for vid in vid_files:
        # Open next video file
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print("Error opening file {}".format(vid))
        filename = utils.get_filename(vid)
        width, height = det.get_video_information(cap, filename)
        # Enter video processing
        processed_frames = 0
        start_time_vid = cv2.getTickCount()
        while cap.isOpened():
            # Read next frame
            ret, bgr_frame = cap.read()
            # Check for error while retrieving frame
            if ret:
                #
                # Lane Detection
                #
                # Convert to HLS color space
                hls_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HLS)
                # Convert to binary
                _, binary_frame = cv2.threshold(hls_frame[:, :, 2], 110, 255, cv2.THRESH_BINARY)
                # Blur image
                binary_frame = cv2.GaussianBlur(binary_frame, (3, 3), 0)
                # Detect edges
                edge_frame = det.get_sobel_edge_detection(binary_frame)

                # Mask image
                # ToDo calculate mean of street in front of car
                # calculate mean of sky -> start mask when hitting the road (add max of 0.5 image and apply threshold)
                # edge_frame[0:int(height/3), :] = (0, 0, 0)

                # Display results
                processed_frames += 1
                cv2.imshow("original_id", bgr_frame)
                cv2.setWindowTitle("original_id", filename)
                cv2.imshow("lanes_id", edge_frame)
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
            """
                .format(int(1 / det.get_passed_time(
                start_time_vid, cv2.getTickCount(), processed_frames))
                        ))
        # Release video file
        cap.release()
        # Check for manual end command
        if end_application:
            break
    # Total application runtime
    print("Total processing time: {}s".format(det.get_passed_time(start_time_app, cv2.getTickCount())))
    # Close Window
    cv2.destroyAllWindows()


#
# Run as script
if __name__ == "__main__":
    main()
