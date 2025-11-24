import cv2
from facemesh_processor import FaceMeshProcessor

# Initialize FaceMesh processor
processor = FaceMeshProcessor()

# Path to the input image
image_path = "input.jpg"

# Load the image from disk (BGR format by default)
image = cv2.imread(image_path)

# Process the image
processed = processor.process_frame(image)

# Save the processed output image
cv2.imwrite("output.jpg", processed)

print("Saved output.jpg")
