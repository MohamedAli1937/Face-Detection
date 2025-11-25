import cv2
import mediapipe as mp
import numpy as np

# --------------------------------
# CLASS TO EXTRACT FACE LANDMARKS
# --------------------------------
class FaceLandmarkExtractor:
    def __init__(self, max_faces=1, static_image_mode=True, refine_landmarks=True, min_detection_confidence=0.5):
        """
        Initialize the FaceLandmarkExtractor with Mediapipe Face Mesh.

        Parameters:
        - max_faces: Maximum number of faces to detect.
        - static_image_mode: True for processing single images; False for video stream.
        - refine_landmarks: True to get more detailed landmarks (e.g., iris).
        - min_detection_confidence: Minimum confidence value for face detection.
        """
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence
        )

    def extract(self, image, normalize=True):
        """
        Extract 3D facial landmarks from an image.

        Parameters:
        - image: Input image (BGR format).
        - normalize: If True, shift coordinates so the minimum is at 0 for all axes.

        Returns:
        - Flattened list of 3D coordinates: [x0, y0, z0, x1, y1, z1, ...]
          If no face is detected, returns an empty list.
        """

        # Get image height and width
        h, w = image.shape[:2]

        # Scale image to maximum dimension 512
        scale = 512 / max(h, w)
        img_resized = cv2.resize(image, (int(w * scale), int(h * scale)))

        # Convert BGR to RGB as Mediapipe requirement
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Run face mesh detection
        results = self.mp_face_mesh.process(img_rgb)

        # If no faces detected, return empty list
        if not results.multi_face_landmarks:
            return []

        # Take landmarks of the first detected face
        landmarks = results.multi_face_landmarks[0].landmark

        # Convert landmarks to a NumPy array (shape: num_landmarks x 3)
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        if normalize:
            # Normalize coordinates to start from 0 (relative to bounding box)
            coords -= coords.min(axis=0)

        # Flatten to 1D list for easier use in ML models
        return coords.flatten().tolist()