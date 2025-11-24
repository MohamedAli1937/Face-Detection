import cv2
from facemesh_processor import FaceMeshProcessor

# Initialize FaceMesh processor
processor = FaceMeshProcessor()

# Open the input video file
cap = cv2.VideoCapture("input.mp4")

# Read video properties needed to save the output
fps = cap.get(cv2.CAP_PROP_FPS)             # Frames per second
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Video width
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Video height

# Create video writer to save the processed output
writer = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Process the video frame by frame
    frame = processor.process_frame(frame)

    # Write processed frame into output video
    writer.write(frame)

cap.release()
writer.release()
print("Saved output.mp4")

