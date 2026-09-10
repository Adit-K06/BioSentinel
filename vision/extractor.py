# FILE: vision/extractor.py
#
# PURPOSE:
# Real, working OpenCV pipeline that extracts the foam boundary height
# (in pixels) from a camera frame, and tracks how it changes over time
# (dh/dt). This code is genuine computer vision — only the input frames
# are synthetic; the pipeline itself would work unchanged on a real
# camera feed.
#
# WHAT IT WILL CONTAIN:
# - A function that takes a frame, converts to grayscale, applies a
#   threshold, finds contours, and picks the largest contour to
#   determine the boundary's pixel height.
# - A tracker class that stores a history of (timestamp, height) pairs
#   and computes the rate of change (dh/dt) between consecutive frames.
#
# WHAT IT WILL DO:
# Feeds boundary height + dh/dt as real-time visual features into the
# Job 1 (foam) ML model, alongside the telemetry features (RQ, RPM,
# airflow, pressure).
