import cv2
from facemesh_processor import FaceMeshProcessor

# Initialize FaceMesh processor
processor = FaceMeshProcessor()

# Open webcam stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Process the frame
    frame = processor.process_frame(frame)

    # Display processed frame
    cv2.imshow("Webcam FaceMesh", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
