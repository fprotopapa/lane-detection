##########################################################
# get_coordinates.py
#
# SPDX-FileCopyrightText: Copyright 2021 Fabbio Protopapa
#
# SPDX-License-Identifier: MIT
#
# Select coordinates from image
#
# ########################################################
#
# importing the module
import cv2
import numpy as np

coordinates = []

def set_polygon(image):
    print("Left click to select point, click 'r' to restart selection, 'q' to quit.")
    print("left_bottom, left_top, right_top, right_bottom")
    roi_image = image.copy()
    # setting mouse handler for the image
    # and calling the click_event() function
    window_name = 'setPolygon r:reset q:quit'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event, roi_image)
    #cv2.imshow(window_name,image)
    while(True):
        cv2.imshow(window_name, roi_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            roi_image = image.copy()
            cv2.setMouseCallback(window_name, click_event, roi_image)
            coordinates.clear()
            print("Clear selection.")
            print("left_bottom, left_top, right_top, right_bottom")
        elif key == ord("q"):
            if len(coordinates) >= 4:
                break
            else:
                print('First select four points.')
        # Draw ROI
        num_points = len(coordinates)
        if num_points == 2:
            cv2.circle(roi_image, coordinates[1], radius=1, color=(255, 0, 0), thickness=-1)
            cv2.line(roi_image, coordinates[0], coordinates[1], color=(255, 0, 0), thickness=1)
        elif num_points == 3:
            cv2.circle(roi_image, coordinates[2], radius=1, color=(255, 0, 0), thickness=-1)
            cv2.line(roi_image, coordinates[1], coordinates[2], color=(255, 0, 0), thickness=1)
        elif num_points == 4:
            cv2.circle(roi_image, coordinates[3], radius=1, color=(255, 0, 0), thickness=-1)
            cv2.line(roi_image, coordinates[2], coordinates[3], color=(255, 0, 0), thickness=1)
            cv2.line(roi_image, coordinates[3], coordinates[0], color=(255, 0, 0), thickness=1)
        elif num_points == 1:
            cv2.circle(roi_image, coordinates[0], radius=1, color=(255, 0, 0), thickness=-1)

    cv2.destroyAllWindows()
    return [coordinates[0], coordinates[1], coordinates[2], coordinates[3]]


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, image):
    # left_bottom, left_top, right_top, right_bottom
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print('col: {}, row: {}'.format(x, y))
        # Save points
        coordinates.append((x, y))
        

# Run as script
if __name__=="__main__":
    import utils
    image_folder_name = 'input_image'
    image_paths = utils.get_list_of_input_files(image_folder_name, "jpg")
    # reading the image
    image = cv2.imread(image_paths[0])

    cords = set_polygon(image)
    print(cords)
    
