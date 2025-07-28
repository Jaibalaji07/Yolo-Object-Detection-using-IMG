import cv2
import numpy as np

# Load YOLO model
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

# Read the class names from coco.names
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]
    
layer_names = yolo.getLayerNames()

output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

# Define colors for the rectangles and text
colorRed = (0,0,255)   #Blue,Green,Red
colorGreen = (0,255,0)

# #Loading Images
name = "new6.jpg"
img = cv2.imread(name)
# Get the dimensions of the image
height, width, channels = img.shape

# Prepare the image for the YOLO model
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set the input for the model
yolo.setInput(blob)

# Get the output predictions from the model
outputs = yolo.forward(output_layers)
# Initialize lists to hold detected class IDs, confidence scores, and bounding boxes
class_ids = []
confidences = []
boxes = []

# Process each output from the model
for output in outputs:
    for detection in output:
        scores = detection[5:]        # Confidence scores for each class
        #detection = [0.5, 0.5, 0.2, 0.3, 0.9, 0.2, 0.8, 0.1]
        #scores = [0.2, 0.8, 0.1] # (car, person, tree)
        class_id = np.argmax(scores)        # Get the class with the highest score
        #The highest score is 0.8 (for person), so class_id will be 1.
        confidence = scores[class_id]
        #scores = [0.2, 0.8, 0.1]  # (0.2 for car, 0.8 for person, 0.1 for tree)
        if confidence > 0.5:
            ## Calculate the coordinates and size of the bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Calculate the top-left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
# Apply Non-Maximum Suppression to eliminate redundant overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        
        cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
        cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 2, colorRed, 2)

#
cv2.imwrite("Output/output6.jpg",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
