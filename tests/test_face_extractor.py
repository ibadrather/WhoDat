import os
import sys
import numpy as np
import pytest
import cv2
# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)


# Import the FaceExtractor class
from Common.FaceExtractor import FaceExtractor

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def test_image() -> Any:
    # Load a test image
    test_image_path = "/home/ibad/Desktop/WhoDat/tests/test_media/faces.jpg"
    test_image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
    return test_image

@pytest.fixture
def test_image_without_faces() -> Any:
    # Load a test image
    test_image_without_faces = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    return test_image_without_faces


@pytest.fixture
def face_extractor() -> FaceExtractor:
    # Create a FaceExtractor object
    return FaceExtractor()

def test_extract_faces_from_image(face_extractor: FaceExtractor, test_image: Any) -> None:
    """Test the extract_faces_from_image method."""

    # Extract faces from the image
    faces = face_extractor.extract_faces_from_image(test_image)

    # Assert that the length of the faces list is 0
    assert len(faces) > 0


def test_extract_faces_from_image_without_faces(face_extractor: FaceExtractor, test_image_without_faces: Any) -> None:
    """Test the extract_faces_from_image method with an image without faces."""

    # Extract faces from the image
    faces = face_extractor.extract_faces_from_image(test_image_without_faces)

    # Assert that the length of the faces list is 0
    assert len(faces) == 0


def test_extract_faces_from_video_with_frame_drop(face_extractor: FaceExtractor) -> None:
    """Test the extract_faces_from_video method with frame skipping."""

    # Load a video
    video_path = "/home/ibad/Desktop/WhoDat_Dataset/ibad/VID20221016154143.mp4"

    # Extract faces from the video with a frame drop of 10 frames
    faces = face_extractor.extract_faces_from_video(video_path, frame_skip=50)

    # Assert that the length of the faces list is greater than 0
    assert len(faces) > 0


def test_extract_faces_from_video_with_invalid_path(face_extractor: FaceExtractor) -> None:
    """Test the extract_faces_from_video method with an invalid video path."""

    # Load a video with an invalid path
    video_path = Path("invalid_path.mp4")

    # Attempt to extract faces from the video
    with pytest.raises(Exception):
        faces = face_extractor.extract_faces_from_video(video_path)


def test_extract_faces_from_video_with_no_faces(face_extractor: FaceExtractor) -> None:
    """Test the extract_faces_from_video method with a video containing no faces."""

    # Load a video with no faces
    video_path = "/home/ibad/Desktop/WhoDat/tests/media/video_without_faces.mp4"

    # Extract faces from the video
    faces = face_extractor.extract_faces_from_video(video_path)

    # Assert that the length of the faces list is 0
    assert len(faces) == 0

