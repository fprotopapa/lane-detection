##########################################################
# utils.py
#
# SPDX-FileCopyrightText: Copyright 2021 Fabbio Protopapa
#
# SPDX-License-Identifier: MIT
#
# Helper functions
#
# ########################################################
#
# Import libraries
import glob
import os
import platform

#
import cv2


#
#
# Print System and Software version
def print_version_information():
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


#
# Calculate passed time between two operations per frame
def get_passed_time(start_tick, end_tick, frames=1):
    return ((end_tick - start_tick) / cv2.getTickFrequency()) / frames


#
# Get list of media input files
def get_list_of_input_files(folder_name, file_ext="mp4"):
    return sorted(glob.glob(os.path.join(os.path.join(os.getcwd(), folder_name), "*." + file_ext)))


#
# Get filename from path
def get_filename(path):
    _, filename = os.path.split(path)
    return filename
