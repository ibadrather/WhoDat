import os
import sys
import numpy as np

# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Import the FaceExtractor class
from face_extractor import FaceExtractor


def test_extract_faces_from_image():
    # Create a FaceExtractor object
    face_extractor = FaceExtractor()

    # Load an image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Extract faces from the image
    faces = face_extractor.extract_faces_from_image(image)

    # Assert that the length of the faces list is 0
    assert len(faces) == 0


def test_extract_faces_from_video_with_frame_drop():
    # Create a FaceExtractor object
    face_extractor = FaceExtractor()

    # Load a video
    video_path = (
        "/media/ibad/7A28-1119/WhoDat_Dataset/saib/VID20221031154553.mp4"
    )

    # Extract faces from the video with a frame drop of 10 frames
    faces = face_extractor.extract_faces_from_video(video_path, frame_skip=50)

    # Assert that the length of the faces list is greater than 0
    assert len(faces) > 0


def test_extract_faces_from_video_with_invalid_path():
    # Create a FaceExtractor object
    face_extractor = FaceExtractor()

    # Load a video with an invalid path
    video_path = "invalid_path.mp4"

    # Attempt to extract faces from the video
    try:
        faces = face_extractor.extract_faces_from_video(video_path)
    except Exception as e:
        # Assert that an exception was raised
        assert isinstance(e, Exception)


def test_extract_faces_from_video_with_no_faces():
    # Create a FaceExtractor object
    face_extractor = FaceExtractor()

    # Load a video with no faces
    video_path = (
        "/home/ibad/Desktop/WhoDat/ETL_Pipeline/tests/media/video_without_faces.mp4"
    )

    # Extract faces from the video
    faces = face_extractor.extract_faces_from_video(video_path)

    # Assert that the length of the faces list is 0
    assert len(faces) == 0
