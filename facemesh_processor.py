import cv2
import mediapipe as mp


class FaceMeshProcessor:
    def __init__(self, max_faces=1):


        # Initialize Mediapipe modules
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        # Create FaceMesh object for static images
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,      # Static image mode (not webcam)
            max_num_faces=max_faces,      # Maximum number of faces to detect
            refine_landmarks=True,        # More precise landmarks (iris)
            min_detection_confidence=0.5  # Minimum confidence for detection
        )

    def process_frame(self, frame):

        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run FaceMesh
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:

                # Full tessellation
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lm,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    connection_drawing_spec=self.mp_styles.get_default_face_mesh_tesselation_style()
                )

                # Face contour
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lm,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style()
                )

                # Iris
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lm,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    connection_drawing_spec=self.mp_styles.get_default_face_mesh_iris_connections_style()
                )

        return frame