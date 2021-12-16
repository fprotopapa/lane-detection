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
# Default ROI
default_vertices = [(360, 630), (568, 499), (847, 494), (1017, 623)]
# White color mask threshold
white_lower = [0, 210, 0]
white_upper = [255, 255, 255]
# Yellow color mask threshold
yellow_lower = [10, 0, 110]
yellow_upper = [40, 255, 255]
# Drawing configuration
draw_area = True
road_color = (204, 255, 153)
lane_color = (0, 0, 255)
lane_thickness = 30
# Frame width and height
width = 1280
height = 720
# Sliding window
n_windows = 9
margin = 100
px_threshold = 50
