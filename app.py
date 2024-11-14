import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('runs/detect/train/weights/best.pt')  # Replace with your trained model path

# Initialize the webcam
cap = cv2.VideoCapture(0)  # '0' is the default camera. Change if you have multiple cameras

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLO model on the frame
    results = model.predict(source=frame, conf=0.5)  # Adjust confidence threshold if needed

    # Loop over the detections and draw bounding boxes
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # top-left (x1, y1) and bottom-right (x2, y2)
            conf = box.conf[0]  # confidence score
            cls = box.cls[0]  # class index

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time YOLO Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
