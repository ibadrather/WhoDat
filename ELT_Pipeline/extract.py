"""
This module contains the extract function and its supporting functions.

The following steps are performed in the extract function:

1. Get a list of all the folders present in the data directory. The name of each folder is the name of the person or the class.
2. For each folder, get a list of all the files present in the folder. We are interested in video and image files.
3. Extract faces from videos and images of each person/class and save them in a folder with the name of the person/class.
4. Repeat steps 1-3 for all the folders present in the data directory.
"""

import os
import cv2
from tqdm import tqdm
from Common.FaceExtractor import FaceExtractor


def extract_faces_from_directory(
    directory_path: str, output_directory_path: str, method: str = "dlib"
) -> None:
    """
    Extract faces from images and videos in a directory using the selected face detection method.
    Saves the extracted faces to a folder with the name of the person/class in the specified output directory.

    Args:
        directory_path (str): The path to the directory containing the images and videos.
        output_directory_path (str): The path to the directory where the extracted faces should be saved.
        method (str): The face detection method to use. Available options are 'dlib' and 'opencv'.
            Default is 'dlib'.
    """
    # Create a FaceExtractor object
    face_extractor = FaceExtractor(method=method)

    # Iterate over the folders in the directory
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)

        # Check if the path is a directory
        if os.path.isdir(folder_path):
            print(f"Extracting faces from {folder_name}...")

            # Create a folder to save the extracted faces
            output_folder = os.path.join(output_directory_path, f"{folder_name}")
            os.makedirs(output_folder, exist_ok=True)

            # Iterate over the files in the folder
            # tqdm is used to display a progress bar

            for file_name in tqdm(os.listdir(folder_path), leave=False):
                file_path = os.path.join(folder_path, file_name)

                # Check if the path is a file
                if os.path.isfile(file_path):
                    # Get the file extension
                    file_ext = os.path.splitext(file_name)[1]

                    # Extract faces from images and videos
                    if file_ext in [".jpg", ".jpeg", ".png"]:
                        image = cv2.imread(file_path)
                        faces = face_extractor.extract_faces_from_image(image)
                    elif file_ext in [
                        ".mp4",
                        ".avi",
                        ".mov",
                        ".mkv",
                        ".flv",
                        ".wmv",
                        ".webm",
                    ]:
                        faces = face_extractor.extract_faces_from_video(
                            file_path, frame_skip=15, size=(128, 128)
                        )

                    # Save the extracted faces to the output folder
                    for i, face in enumerate(faces):
                        output_path = os.path.join(
                            output_folder, f"{file_name}_{i}.jpg"
                        )
                        cv2.imwrite(output_path, face)


def extract(input_dir: str, output_dir: str, method: str) -> None:
    extract_faces_from_directory(input_dir, output_dir, method)
