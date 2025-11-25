# 🎭 **Face Mesh Processor**

This repository demonstrates **real-time facial landmark detection** using **MediaPipe Face Mesh**. It allows processing:

- Static images
- Video files
- Live webcam feed

The project uses **MediaPipe**, a Google library for cross-platform, real-time face, hand, and body tracking.  

**Link :** [Official MediaPipe Face Mesh documentation](https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html)



## 🧩 **Repository Structure**
```bash
├── README.md
├──requirements.txt       
├── facemesh_processor.py # Shared processor class for face landmarks
├── image_process.py      # Script to process a single image
├── video_process.py      # Script to process a video file
├── webcam_process.py     # Script to process webcam feed
├── extractLandmarks.py   # Landmarks Extractor from face 
├── landmarks2txt.py      # Landmarks → txt
├── exemple_image.jpg/    # Example input image
└── exemple_video.mp4/    # Example input video
```


## 📦 **Installation**

Ensure **Python 3.7+** is installed. Install the dependencies:

```bash
pip install opencv-python mediapipe 
```
## 🧠 **How It Works**
### 1️⃣ **Face Mesh Processor** 

The FaceMeshProcessor class contains the main logic:

- Converts images to RGB format (required by MediaPipe).

- Detects facial landmarks using `mp.solutions.face_mesh`.

- Draws landmarks (tessellation, contours, and irises) on images.

- Returns processed images for display or saving.
### 2️⃣ **Image Processing**

- Reads a single image.

- Uses FaceMeshProcessor to detect and draw landmarks.

- Saves the annotated image.
```
python image_process.py
```

<img src="https://github.com/MohamedAli1937/Face-Detection/blob/384dfdba76532c9fa378835d7bc7c3c5722bc62c/exemple_image.png" alt="Input Image" width="400"/>

### 3️⃣ **Video Processing**

- Reads video frame by frame.

- Uses FaceMeshProcessor for each frame.

- Writes processed frames to a new video file.
```
python video_process.py
```
![Watch Output Video](https://github.com/user-attachments/assets/4dcdffec-ab97-49a6-b539-38d8030e69fd)
### 4️⃣ **Webcam Processing**

- Captures live feed from the webcam.

- Processes each frame in real-time.

- Displays annotated frames.
```
python webcam_process.py
```
## 🕹️ **Interest of Face Landmarks**
### 1️⃣ **`extractLandmarks.py`**

**Purpose:**

- Extract 3D facial landmarks from images using Mediapipe Face Mesh.

- Provides a convenient class FaceLandmarkExtractor that returns the landmarks in a flattened array format.

- Can be used for single images or video frames.

**Key Functions:**
```python
class FaceLandmarkExtractor:
    def __init__(...):
        # Initialize Mediapipe Face Mesh
        pass

    def extract(self, image, normalize=True):
        # Extract facial landmarks from an image
        # Returns a flat 1D list: [x0, y0, z0, x1, y1, z1, ...]
        # If normalize=True, coordinates are shifted to start from 0
        pass
```
**How to use:**
```python
from extractLandmarks import FaceLandmarkExtractor
import cv2

extractor = FaceLandmarkExtractor()

image = cv2.imread("happy_face.jpg")
landmarks = extractor.extract(image)  # landmarks is a list of 1404 floats (468 points x 3)
```

→ Landmarks can now be used as input features for an **ML model.**
### 2️⃣ **`landmarks2txt.py`**
**Purpose:**

- Loops through a dataset of images organized by emotion folders.

- Uses FaceLandmarkExtractor to extract landmarks for all images.

- Adds the emotion label to the end of each landmark vector.

- Saves the result as a text file `data.txt` for training ML models.

**Expected folder structure:**
```bash
data/
    happy/
        img1.jpg
        img2.jpg
    sad/
        img1.jpg
        img2.jpg
```
- Each row in `data.txt` has **1405 values → 1404 landmarks + 1 label.**
### **3️⃣ Workflow to Train Emotion Model**

1. Collect **dataset** of faces organized by emotion.

2. Run `landmarks2txt.py` → generates `data.txt`.

3. Load `data.txt` → features = landmarks, labels = emotion.

4. Train **ML model** (Dense NN or other) on this data.

**For live prediction:**

Capture frame → extract landmarks → feed to model → get emotion.

## 📝 **Notes**

- Supported image formats: `.jpg` `.png`

- Supported video formats: `.mp4` `.avi`

Make sure you replace `input.jpg` and `input.mp4` with your own files.
