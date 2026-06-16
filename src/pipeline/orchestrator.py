"""Pipeline orchestrator — runs all modules in reference workflow order."""

# Reference order:
# Image Processing -> Lane Detection -> Object Detection ->
# Traffic Sign Recognition -> Traffic Signal Detection ->
# Semantic Segmentation -> Decision Support
