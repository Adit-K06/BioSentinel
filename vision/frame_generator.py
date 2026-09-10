# FILE: vision/frame_generator.py
#
# PURPOSE:
# Generates fake "camera" images of the bioreactor's foam boundary,
# since no real bioreactor-foam video dataset exists publicly. This
# stands in for a physical camera feed.
#
# WHAT IT WILL CONTAIN:
# - A function that draws a froth boundary line at a given simulated
#   foam height, on a blank frame of a given width/height.
# - Bubble-texture noise and simple lighting variation added on top,
#   so the frame looks visually plausible rather than a flat line.
#
# WHAT IT WILL DO:
# Takes the current simulated foam height (from simulator/foam_model.py)
# and returns an image (numpy array) representing what a camera pointed
# at the bioreactor would roughly see. This image is then fed into
# vision/extractor.py for the OpenCV processing step.
