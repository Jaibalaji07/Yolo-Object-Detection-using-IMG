# yolo_objectDetection_imagesCPU
YOLO Object Detection on Images on a CPU

This project implements robust object detection on images using the YOLOv3 (You Only Look Once) deep learning model with OpenCV. It features a ready-to-use Python script that detects and labels multiple common object classes—such as people, vehicles, animals, household items, and more—in provided images. The system includes:

Automatic detection and labeling of 80 everyday object types (as defined by the COCO dataset and .names file).

Bounding box display around each detected object, with a visually unique color assigned per class for easy distinction.

Class detection confidence scores displayed on the image for each label.

Detection summary in the console: The script prints a count of every distinct object type detected in the image.

Flexible file support: Easily swap in other YOLO .cfg, .weights, and .names files to recognize new objects from different datasets or custom models.

Robust compatibility: Handles variations in OpenCV output, preventing common errors across OpenCV versions.
