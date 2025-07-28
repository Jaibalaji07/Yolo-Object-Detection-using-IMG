<<<<<<< HEAD
import cv2
import numpy as np

# Load YOLOv3 model
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names (default: does not include "tree")
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

# Assign a visually distinct color to each class
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the image
name = "new4.jpg"
img = cv2.imread(name)
if img is None:
    raise FileNotFoundError(f"Image file '{name}' not found.")
height, width, channels = img.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo.setInput(blob)
outputs = yolo.forward(output_layers)

class_ids = []
confidences = []
boxes = []

# Parse YOLO's raw outputs
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Standardize 'indexes' for compatibility
if isinstance(indexes, np.ndarray):
    indices = indexes.flatten()
else:
    indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indexes] if len(indexes) else []

if len(indices):
    detected_counts = {}
    for i in indices:
        class_name = classes[class_ids[i]]
        detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
    for i in indices:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    print("Detected object counts:")
    for class_name, count in detected_counts.items():
        print(f"{class_name}: {count}")
else:
    print("No objects detected.")

cv2.imwrite("Output/new8_op.jpg", img)
=======
import cv2
import numpy as np

# Load YOLOv3 model
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names (default: does not include "tree")
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

# Assign a visually distinct color to each class
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the image
name = "new4.jpg"
img = cv2.imread(name)
if img is None:
    raise FileNotFoundError(f"Image file '{name}' not found.")
height, width, channels = img.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo.setInput(blob)
outputs = yolo.forward(output_layers)

class_ids = []
confidences = []
boxes = []

# Parse YOLO's raw outputs
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Standardize 'indexes' for compatibility
if isinstance(indexes, np.ndarray):
    indices = indexes.flatten()
else:
    indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indexes] if len(indexes) else []

if len(indices):
    detected_counts = {}
    for i in indices:
        class_name = classes[class_ids[i]]
        detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
    for i in indices:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    print("Detected object counts:")
    for class_name, count in detected_counts.items():
        print(f"{class_name}: {count}")
else:
    print("No objects detected.")

cv2.imwrite("Output/new8_op.jpg", img)
>>>>>>> 485f116aab6b12c687063ad278dd3647358bfa3c
