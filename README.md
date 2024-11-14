# Advanced Object Detection Model using YOLOv8 🚀

This project demonstrates an advanced object detection model leveraging YOLOv8's capabilities for high-precision object detection and segmentation. The model can identify and classify multiple objects in images and videos in real time, with emphasis on achieving high accuracy and adaptability across diverse scenarios and lighting conditions.

## Project Overview 🌟

- **Goal**: To develop a powerful object detection and segmentation model that performs reliably across various environments.
- **Model**: YOLOv8 segmentation model for accurate detection using polygons/masks instead of traditional bounding boxes.
- **Applications**: Real-time detection from webcam, contour-based non-rectangular detection, high-precision object tracking.

## Key Features ✨

- **Real-Time Detection**: Detects objects in real time using a webcam.
- **Segmentation Support**: Uses YOLOv8 segmentation model to generate object-fitted polygons.
- **Contour Detection**: Adds flexibility by allowing non-rectangular detection within bounding boxes.
- **Customizable Confidence Threshold**: Fine-tune detection confidence to adjust sensitivity.

## Prerequisites 📦

- Python 3.8 or higher
- Install required packages:
  ```bash
  pip install ultralytics opencv-python
  ```

## Setup Instructions 🛠️

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/advanced-object-detection-yolov8.git
   cd advanced-object-detection-yolov8
   ```

2. **Download YOLOv8 Segmentation Model**:
   - You can use pre-trained models like `yolov8m-seg.pt` or train your own custom model.
   - Place the model file in your project directory and update the model path in the code.

3. **Prepare Your Dataset**:
   - Organize your dataset with images in an `images` folder and corresponding labels in a `labels` folder.
   - Each label file should contain annotations for the object locations in each image.

4. **Training Your Model (Optional)**:
   - Train YOLOv8 with custom data by modifying the following code:
     ```python
     from ultralytics import YOLO
     model = YOLO('yolov8m-seg.pt')
     model.train(data="path/to/data.yaml", epochs=50, batch=16, imgsz=640)
     ```

## Real-Time Detection with Segmentation 💻

Use the following script to detect objects in real time using a webcam:

```python
import cv2
from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO("path/to/your/trained_model.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLO model on the frame
    results = model.predict(source=frame, task='segment', conf=0.5)

    # Draw polygons on the frame
    for result in results:
        for segment in result.masks:
            points = segment.xy[0]
            cv2.polylines(frame, [points.astype(int)], isClosed=True, color=(255, 0, 255), thickness=2)

    # Show frame with detections
    cv2.imshow("Real-Time YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Contour-Based Detection 🖼️

For non-segmentation YOLO models, use contour detection within bounding boxes to get custom shapes:

```python
# Code snippet to detect contours
# Refer to the previous response for full code
```

## Customization Options 🛠️

- **Confidence Threshold**: Adjust the `conf` parameter to control detection sensitivity.
- **Segmentation vs. Bounding Boxes**: Use segmentation models for object-fitted shapes or contour detection for non-rectangular detection.

## License 📄

This project is licensed under the MIT License.

