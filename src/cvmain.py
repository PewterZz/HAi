import cv2
from ultralytics import RTDETR


cap = cv2.VideoCapture(0)  # 0 for the default camera
model = RTDETR("rtdetr-l.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    # Perform detection
    results = model(frame)

    if results:
        # Access the first detection
        first_detection = results[0]

        # Get the bounding boxes, class IDs, and confidences
        boxes = first_detection.boxes.xyxy
        class_ids = first_detection.boxes.cls
        confidences = first_detection.boxes.conf

        # Draw bounding boxes and labels
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
            conf = confidences[i].item()  # Convert tensor to float
            cls_id = class_ids[i].item()  # Convert tensor to integer
            label = f'{model.names[cls_id]} {conf:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('RTDETR Real-Time Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
