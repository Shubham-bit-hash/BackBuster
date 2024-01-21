# import cv2
# import mediapipe as mp
# import numpy as np

# mp_drawing = mp.solutions.drawing_utils
# mp_selfie_segmentation = mp.solutions.selfie_segmentation

# BG_COLOR = (128, 10, 128)
# cap = cv2.VideoCapture(0)

# def change_background(bg_images, current_bg_index):
#     return cv2.resize(bg_images[current_bg_index % len(bg_images)], (640, 480))

# # Initialize the selfie segmentation model
# with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
#     bg_images = [
#         cv2.imread('path_to_background_image1.jpg'),
#         cv2.imread('path_to_background_image2.jpg'),
#         # Add more background images as needed
#     ]
#     current_bg_index = 0
#     bg_image = change_background(bg_images, current_bg_index)

#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#             continue

#         image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False

#         # Process the image using selfie segmentation
#         results = selfie_segmentation.process(image)

#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Create a condition based on the segmentation mask
#         condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

#         # Update background image if user changes it
#         bg_image = change_background(bg_images, current_bg_index)

#         # Apply the condition to blend the foreground and resized background
#         output_image = np.where(condition, image, bg_image)

#         cv2.imshow('MediaPipe Selfie Segmentation', output_image)

#         key = cv2.waitKey(5) & 0xFF
#         if key == 27:  # Press 'Esc' to exit
#             break
#         elif key == ord('n'):  # Press 'n' to switch to the next background image
#             current_bg_index += 1
#         elif key == ord('i'):  # Press 'i' to input the index of the background image
#             new_index = int(input("Enter the index of the background image: "))
#             if 0 <= new_index < len(bg_images):
#                 current_bg_index = new_index

#     cap.release()
#     cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

BG_COLOR = (128, 10, 128)
cap = cv2.VideoCapture(1)

def change_background(bg_images, current_bg_index):
    background_image = bg_images[current_bg_index % len(bg_images)]
    if background_image is not None and not background_image.size == 0:
        return cv2.resize(background_image, (640, 480))
    else:
        return np.zeros((480, 640, 3), dtype=np.uint8)  # Return a black image if background is invalid

# Initialize the selfie segmentation model
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    bg_images = [
        cv2.imread('./back/office.jpg'),
        cv2.imread('./back/bedroom.jpg'),
        cv2.imread('./back/1.jpeg'),
        cv2.imread('./back/2.jpeg'),
        cv2.imread('./back/bedroom.jpg'),
        # Add more background images as needed
    ]
    current_bg_index = 0
    bg_image = change_background(bg_images, current_bg_index)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image using selfie segmentation
        results = selfie_segmentation.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Create a condition based on the segmentation mask
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

        # Update background image if the user changes it
        bg_image = change_background(bg_images, current_bg_index)

        # Apply the condition to blend the foreground and resized background
        output_image = np.where(condition, image, bg_image)

        cv2.imshow('BackBuster', output_image)
        # cv2.imshow('orginal image',image)
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # Press 'Esc' to exit
            break
        elif key == ord('n'):  # Press 'n' to switch to the next background image
            current_bg_index += 1
        elif key == ord('i'):  # Press 'i' to input the index of the background image
            new_index = int(input("Enter the index of the background image: "))
            if 0 <= new_index < len(bg_images):
                current_bg_index = new_index

    cap.release()
    cv2.destroyAllWindows()
