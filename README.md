# BackBuster
Background removal and change the background during live camera feed.
# Real-Time Background Switching with MediaPipe Selfie Segmentation

## Overview

This project utilizes computer vision techniques to enable real-time background switching in a live video feed. It leverages the MediaPipe library for selfie segmentation, allowing users to seamlessly change their background during video capture. The project is implemented in Python using OpenCV and MediaPipe.

## Features

- Real-time background switching using selfie segmentation.
- User-friendly interface for interactive control.
- Predefined set of background images with the ability to add more.
- Dynamic adaptation to changes in the background image set.
- Supports keyboard inputs for background switching and user input.

## Prerequisites

- Python 3.x
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/Shubham-bit-hash/BackBuster/.git
    ```

2. open the folder where the files downloaded:

    ```bash
    downloads/BackBuster/
    ```

3. Run the application:

    ```bash
    python main.py
    ```

4. Use the following keyboard inputs:

    - Press 'Esc' to exit the application.
    - Press 'n' to switch to the next background image.
    - Press 'i' to input the index of the background image.

## Adding More Background Images

1. Add your background images to the `./back/` directory.
2. Update the `bg_images` list in `main.py` with the new image file paths.

```python
bg_images = [
    cv2.imread('./back/office.jpg'),
    cv2.imread('./back/bedroom.jpg'),
    cv2.imread('./back/1.jpeg'),
    cv2.imread('./back/2.jpeg'),
    # Add more background images as needed
]
