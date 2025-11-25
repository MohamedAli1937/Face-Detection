import os
import cv2
import numpy as np
from extractLandmarks import FaceLandmarkExtractor

# --------------------------------
# LANDMARKS ARRAY -> SAVE TO TXT
# --------------------------------
'''
This script reads images from folders named after emotions,
extracts face landmarks for each image, appends the emotion label,
and saves the dataset to a text file.

Directory structure : 
./data/
    happy/
        img1.jpg
        img2.jpg
    sad/
        img1.jpg
        img2.jpg
'''

extractor = FaceLandmarkExtractor()  # Initialize Mediapipe face landmark extractor
data_dir = './data'                  # Directory containing emotion subfolders

output = []                          # Will store all landmarks + labels

# Loop through each emotion folder
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    emotion_folder = os.path.join(data_dir, emotion)

    # Loop through each image in the emotion folder
    for image_name in os.listdir(emotion_folder):
        image_path = os.path.join(emotion_folder, image_name)

        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            continue                  # Skip if image cannot be read

        # Extract facial landmarks
        face_landmarks = extractor.extract(image)

        # Check if landmarks length matches expected number (468 points x 3 = 1404)
        if len(face_landmarks) == 1404:
            # Append the emotion index as the label
            face_landmarks.append(int(emotion_indx))

            # Add this sample to the output list
            output.append(face_landmarks)

# Convert output list to NumPy array and save to text file
np.savetxt('data.txt', np.asarray(output))
print(f"Saved {len(output)} samples to data.txt")