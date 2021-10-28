##########################################################
# lane_detection.py
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
import os
import platform
import glob
#
# User defined configurations
video_folder_name = "input_video"
video_file_format = "*.mp4"

# Print System and Software version
def printVerionInformation():
  print("""
    Python version: {}
    OpenCV: {}
    System: {}
    Machine: {}
    Platform: {}
    ----------------------------------------------------------------------------------------
    """.format(
  platform.sys.version,
  cv2.__version__,
  platform.system(),
  platform.machine(),
  platform.platform()
  ))

# Print Video Information, return video resolution (width, height)
def getVideoInformation(cap, filename):
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  print("""
    File:     {}
    FPS:      {}
    # Frames: {}
    Width:    {}
    Height:   {}
    """
  .format(
  filename,
  int(cap.get(cv2.CAP_PROP_FPS)),
  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
  width,
  height
  ))
  return (width, height)

# Calculate passed time between two operations per frame
def getPassedTime(startTime, endTime, frames=1):
  return ((endTime - startTime)/cv2.getTickFrequency())/frames

def getFPS(time):
  return 
# Main application
def main():
  # General informations
  printVerionInformation()
  # Build paths for media input
  current_dir = os.getcwd()
  vid_path = os.path.join(current_dir, video_folder_name)
  # Search for avaible video files
  vid_files = sorted(glob.glob(os.path.join(vid_path, video_file_format)))
  # For manual application exit
  end_application = False
  # Application start time
  starttime_app = cv2.getTickCount()
  # Loop through found video files
  for vid in vid_files:
    # Open next video file
    cap = cv2.VideoCapture(vid)
    if (cap.isOpened()== False):
      print("Error opening file {}".format(vid))
    _, filename = os.path.split(vid)
    width, height = getVideoInformation(cap, filename)
    # Enter video processing
    processed_frames = 0
    starttime_vid = cv2.getTickCount()
    while(cap.isOpened()):
      # Read next frame
      ret, frame = cap.read()
      # Check for error while retrieving frame
      if ret == True:
        #
        # Lane Detection
        #

        #
        #
        #
        processed_frames += 1
        # Display results
        cv2.imshow("original_id" ,frame)
        cv2.setWindowTitle("original_id", filename)  
        # 'ESC' to skip to next file and 'q' to end application
        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == 27: 
          break
        elif pressedKey == ord('q'):
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
    .format(int(1/getPassedTime(
      starttime_vid, cv2.getTickCount(), processed_frames))
    ))
    # Release video file  
    cap.release()
    # Check for manual end command
    if end_application:
      break
  # Total application runtime
  print("Total processing time: {}s".format(getPassedTime(starttime_app, cv2.getTickCount())))
  # Close Window
  cv2.destroyAllWindows()
#
# Run as script
if __name__ == "__main__":
  main()
