##########################################################
# config.py
#
# SPDX-FileCopyrightText: Copyright 2021 Fabbio Protopapa
#
# SPDX-License-Identifier: MIT
#
# Configuration data
#
# ########################################################
#
# White color mask threshold
white_lower = [0, 210, 0]
white_upper = [255, 255, 255]
# Yellow color mask threshold
yellow_lower = [10, 0, 110]
yellow_upper = [40, 255, 255]
# Drawing configuration
draw_area = True
road_color = (204, 255, 153)
l_lane_color = (0, 0, 255)
r_lane_color = (255, 0, 0)
lane_thickness = 30
# Frame width and height
width = 1280
height = 720
# Sliding window
n_windows = 9
margin = 100
nb_margin = 100
px_threshold = 50
radii_threshold = 6
# Conversion pixel to meter
px_to_m_y = 30/720 # meters per pixel in y dimension
px_to_m_x = 3.7/700 # meters per pixel in x dimension
# Lanes and poly
min_lane_dis = 600
poly_thr_a = 0.001
poly_thr_b = 0.4
poly_thr_c = 150
# Average
queue_len = 20