import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
original_image = cv2.imread('/content/drive/MyDrive/ML_Data/project3_data/motherboard_image.JPEG')

# Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply thresholding
_, thresh = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)


# Use Canny edge detector for edge detection
edges = cv2.Canny(blurred_image, 50, 150)

# Find contours in the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours based on area
min_contour_area = 1000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Create a mask and extract the PCB from the original image
mask = np.zeros_like(gray_image)
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Bitwise AND operation to extract the PCB
extracted_image = cv2.bitwise_and(original_image, original_image, mask=mask)

# Convert the extracted image to RGB
colored_extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2RGB)

# Comparison images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(colored_extracted_image)
plt.title("Extracted Image")

plt.imshow(mask, cmap='gray')
plt.title("Mask Image")
plt.show()

plt.show()

# Step 2

!pip install ultralytics
import os
import numpy as np
import cv2
from ultralytics import YOLO

# Loading the dataset
train_dir = '/content/drive/MyDrive/ML_Data/project3_data/train/images'
eval_dir = '/content/drive/MyDrive/ML_Data/project3_data/evaluation'

# Loading the YOLOv8 model
model = YOLO('/content/drive/MyDrive/ML_Data/project3_data/yolov8s.pt')

# Define hyperparameters
epochs = 150
batch = 272
img_size = 120

# Train the model
model.train(data='/content/drive/MyDrive/ML_Data/project3_data/data.yaml', epochs=epochs, batch=batch, imgsz=img_size)

#Step 3
# Evaluate the model
img1 = cv2.imread(os.path.join(eval_dir, 'ardmega.jpg'))
img2 = cv2.imread(os.path.join(eval_dir, 'arduno.jpg'))
img3 = cv2.imread(os.path.join(eval_dir, 'rasppi.jpg'))

# Run object detection
outputs1 = model.predict(img1)
outputs2 = model.predict(img2)
outputs3 = model.predict(img3)

# Print the detection results
print('Image 1:')
print(outputs1[0])
print('Image 2:')
print(outputs2[0])
print('Image 3:')
print(outputs3[0])

# Analyze the results
img1_detected_components = []
img2_detected_components = []
img3_detected_components = []


for detection in outputs1[0].boxes.data.tolist(): # Access the detection data as a list
    class_id = int(detection[5]) # Get class ID directly 
    confidence = detection[4] # Get confidence score directly
    if confidence > 0.5:
        img1_detected_components.append(class_id)

for detection in outputs2[0].boxes.data.tolist():
    class_id = int(detection[5])
    confidence = detection[4]
    if confidence > 0.5:
        img2_detected_components.append(class_id)

for detection in outputs3[0].boxes.data.tolist():
    class_id = int(detection[5])
    confidence = detection[4]
    if confidence > 0.5:
        img3_detected_components.append(class_id)

print('Image 1: Detected components:', img1_detected_components)
print('Image 2: Detected components:', img2_detected_components)
print('Image 3: Detected components:', img3_detected_components)

# Summarize the model's performance
accuracy = 0
for image in img1_detected_components + img2_detected_components + img3_detected_components:
    accuracy += 1
accuracy /= 3
print('Model accuracy:', accuracy)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/content/drive/MyDrive/ML_Data/project3_data/yolov8s.pt')

# Define the evaluation directory
eval_dir = '/content/drive/MyDrive/ML_Data/project3_data/evaluation'

# Load the test images
img1 = cv2.imread(os.path.join(eval_dir, 'ardmega.jpg'))
img2 = cv2.imread(os.path.join(eval_dir, 'arduno.jpg'))
img3 = cv2.imread(os.path.join(eval_dir, 'rasppi.jpg'))

# Run object detection
outputs1 = model.predict(img1)
outputs2 = model.predict(img2)
outputs3 = model.predict(img3)

# Function to draw bounding boxes
def draw_bounding_boxes(image, outputs):
    for detection in outputs[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = map(int, detection[:6])
        if confidence > 0.5:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'Class {class_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# Draw bounding boxes on the images
img1_with_boxes = draw_bounding_boxes(img1.copy(), outputs1)
img2_with_boxes = draw_bounding_boxes(img2.copy(), outputs2)
img3_with_boxes = draw_bounding_boxes(img3.copy(), outputs3)

# Convert images to RGB for displaying with Matplotlib
img1_rgb = cv2.cvtColor(img1_with_boxes, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2_with_boxes, cv2.COLOR_BGR2RGB)
img3_rgb = cv2.cvtColor(img3_with_boxes, cv2.COLOR_BGR2RGB)

# Display the images with bounding boxes
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img1_rgb)
plt.title("Image 1 with Detected Components")

plt.subplot(1, 3, 2)
plt.imshow(img2_rgb)
plt.title("Image 2 with Detected Components")

plt.subplot(1, 3, 3)
plt.imshow(img3_rgb)
plt.title("Image 3 with Detected Components")

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml

# Load the training results
results_dir = 'runs/detect/train'
confusion_matrix = np.load(f'/content/drive/MyDrive/ML_Data/project3_data/data.yaml/confusion_matrix.npy')
precision_confidence = np.load(f'{results_dir}/precision_confidence.npy')
precision_recall = np.load(f'{results_dir}/precision_recall.npy')

# Plot Normalized Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Blues')
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plot Precision-Confidence Curve
plt.figure(figsize=(10, 8))
plt.plot(precision_confidence[:, 0], precision_confidence[:, 1], marker='o')
plt.title('Precision-Confidence Curve')
plt.xlabel('Confidence Threshold')
plt.ylabel('Precision')
plt.grid(True)
plt.show()

# Plot Precision-Recall Curve
plt.figure(figsize=(10, 8))
plt.plot(precision_recall[:, 0], precision_recall[:, 1], marker='o')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()
