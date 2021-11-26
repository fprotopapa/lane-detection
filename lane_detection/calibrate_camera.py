##########################################################
# calibrate_camera.py
#
# SPDX-FileCopyrightText: Copyright 2021 Fabbio Protopapa
#
# SPDX-License-Identifier: MIT
#
# Use chessboard images to calibrate camera
#
# ########################################################
#
import utils
import os
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import pickle

image_folder_name = 'camera_calibration'
base_path = 'lane_detection'

def main():
    image_paths = utils.get_list_of_input_files(os.path.join(base_path, image_folder_name), "jpg")
    
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    nx = 9 #number of inside corners in x
    ny = 6 #number of inside corners in y

    # Prepare obj points, like (0, 0, 0), (1, 0, 0), (2, 0, 0)....., (7, 5, 0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] =  np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x,y coordinates 

    fig_size = (16, 12)
    plt.figure(figsize=fig_size)
    plot_img = []

    not_found = []
    for path in image_paths:
        image = cv2.imread(path)
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        # Find the Chesse board corners
        ret, corners = cv2.findChessboardCorners(gray_image, (nx, ny), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)  
            
            # Draw and display the corners
            image = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            plot_img.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        else:
            not_found.append(path)
    print("{} images not found. Images at:".format(len(not_found)))
    print(not_found)
    cols = 2
    rows = (len(plot_img) + 1) // cols
    for indx, image in enumerate(plot_img):
        indx += 1
        plt.subplot(rows, cols, indx)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    pickle.dump([objpoints, imgpoints], open(os.path.join(base_path, 'calibration_data', "calibration.p"), "wb" ))

#
# Run as script
if __name__ == "__main__":
    main()
