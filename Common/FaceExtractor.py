import dlib
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Tuple


class FaceExtractor:
    def __init__(self, method="dlib"):
        """
        Initialize the FaceExtractor object with the selected face detection method.

        Args:
            method (str): The face detection method to use. Available options are 'dlib' and 'opencv'.
            Default is 'dlib'.
        """
        self.method = method
        if self.method == "dlib":
            self.detector = dlib.get_frontal_face_detector()
        elif self.method == "opencv":
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        else:
            raise ValueError("Invalid face detection method specified.")

    def extract_faces_from_image(
        self, image: np.ndarray, size: tuple = (128, 128)
    ) -> list[np.ndarray]:
        """
        Extract faces from an image using the selected face detection method.

        Args:
            image (np.ndarray): The image to extract faces from.
            size (tuple): The size to resize each face to. Default is (128, 128).

        Returns:
            list[np.ndarray]: The extracted faces.
        """
        if self.method == "dlib":
            dets = self.detector(image, 1)
            faces = [
                image[det.top() : det.bottom(), det.left() : det.right()]
                for det in dets
            ]
        elif self.method == "opencv":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dets = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            faces = [image[y : y + h, x : x + w] for (x, y, w, h) in dets]
        else:
            raise ValueError("Invalid face detection method specified.")

        # Resize each face to the given size
        faces = [cv2.resize(face, size) for face in faces if face.size > 0]

        return faces

    def extract_faces_and_coordinates_from_image(
        self, image: np.ndarray, size: tuple = (128, 128)
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Extract faces and their bounding box coordinates from an image using the selected face detection method.

        Args:
            image (np.ndarray): The image to extract faces from.
            size (tuple): The size to resize each face to. Default is (128, 128).

        Returns:
            Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]: The extracted faces and their bounding box coordinates.
        """
        coordinates = []
        if self.method == "dlib":
            dets = self.detector(image, 1)
            faces = [
                image[det.top() : det.bottom(), det.left() : det.right()]
                for det in dets
            ]
            coordinates = [
                (det.left(), det.top(), det.right(), det.bottom()) for det in dets
            ]
        elif self.method == "opencv":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dets = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            faces = [image[y : y + h, x : x + w] for (x, y, w, h) in dets]
            coordinates = [(x, y, x + w, y + h) for (x, y, w, h) in dets]
        else:
            raise ValueError("Invalid face detection method specified.")

        # Resize each face to the given size
        faces = [cv2.resize(face, size) for face in faces if face.size > 0]

        return faces, coordinates

    def extract_faces_from_video(
        self, video_path: str, frame_skip: int = 5, size: tuple = (128, 128)
    ) -> list[np.ndarray]:
        """
        Extract faces from a video using the selected face detection method.

        Args:
            video_path (str): The path to the video file.
            frame_skip (int): The number of frames to skip between each frame. Default is 5.
            size (tuple): The size to resize each face to. Default is (128, 128).

        Returns:
            list[np.ndarray]: The extracted faces.
        """
        faces_in_video = []

        try:
            # Open the video
            cap = cv2.VideoCapture(video_path)

            # Get the video's properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Iterate over the frames and extract faces
            for i in tqdm(range(frame_count), desc="Extracting faces from video"):
                if i % frame_skip == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    faces = self.extract_faces_from_image(frame, size=size)
                    faces_in_video.extend(faces)
                else:
                    # Skip the frame
                    cap.grab()

        except cv2.error as e:
            raise Exception(f"Could not open the video: {e}")

        finally:
            # Release resources
            if "cap" in locals() and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()

        return faces_in_video
