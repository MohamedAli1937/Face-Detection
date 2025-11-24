# 🎭 **Face Mesh Processor**

This repository demonstrates **real-time facial landmark detection** using **MediaPipe Face Mesh**. It allows processing:

- Static images
- Video files
- Live webcam feed

The project uses **MediaPipe**, a Google library for cross-platform, real-time face, hand, and body tracking.  

Official MediaPipe Face Mesh documentation: [https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html](https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html)



## 🧩 **Repository Structure**
```bash
├── README.md
├──requirements.txt       
├── facemesh_processor.py # Shared processor class for face landmarks
├── image_process.py      # Script to process a single image
├── video_process.py      # Script to process a video file
├── webcam_process.py     # Script to process webcam feed
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
## 📝 **Notes**

- Supported image formats: `.jpg` `.png`

- Supported video formats: `.mp4` `.avi`

Make sure you replace `input.jpg` and `input.mp4` with your own files.
