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
import matplotlib.pyplot as plt
import numpy as np


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
def get_list_of_input_files(folder_name, file_ext="mov"):
    return glob.glob(os.path.join(os.path.join(os.getcwd(), folder_name), "*." + file_ext))


#
# Get filename from path
def get_filename(path):
    _, filename = os.path.split(path)
    return filename

def imshow_images(images, names=None):
    if names is None or type(names) is not list:
        if type(images) is list:
            names = ['img'+str(num) for num in range(len(images))]
        else:
            if names is None: 
                names = 'img'
            cv2.imshow(names, images)
            cv2.waitKey(0)
            return
    for image, name in zip(images, names):
        cv2.imshow(name, image)
    cv2.waitKey(0)


def close_windows():
    cv2.destroyAllWindows()

#
# Show images with pyplot
def show_images(images, cmap=None, channels=False):
    fig_size = (16, 12)
    plt.figure(figsize=fig_size)
    if channels:
        cols = 3
        rows = len(images)
        off = 0
        for indx, image in enumerate(images):
            indx += off
            for j in range(0, 3):
                plt.subplot(rows, cols, indx + 1 + j)
                plt.imshow(image[:, :, j], cmap='gray')
                plt.axis('off')
            off += 2
    else:
        cols = 2
        rows = (len(images) + 1) // cols
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            cmap = 'gray' if len(image.shape) == 2 else cmap
            if cmap == 'bgr':
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap=None)
            else:
                plt.imshow(image, cmap=cmap)
    plt.tight_layout()
    plt.show()

 
def show_hls_channels(hls_frame):
    cv2.imshow('hue', hls_frame[:, :, 0])
    cv2.imshow('lightness', hls_frame[:, :, 1])
    cv2.imshow('saturation', hls_frame[:, :, 2])

def save_load_np_var(filename, data = None, save = True):
    path = os.path.join('roi', filename + '.npy')
    if save: 
        np.save(path, data)
        return True
    else:
        return np.load(path)