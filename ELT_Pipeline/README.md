
# WhosDat ELT Process

WhosDat is a facial recognition project that utilizes an Extract, Load, and Transform (ELT) process for preparing the dataset. This process involves extracting faces from images and videos, loading the extracted data into a structured format (CSV file), and transforming the data during the data augmentation stage.

## Table of Contents

- [Extract](#extract)
- [Load](#load)
- [Transform](#transform)

## Extract

The extraction process involves detecting and extracting faces from images and videos using a specified face detection method. The extracted faces are then saved into separate folders, each named after the respective class or person.

The main function for the extraction process is `extract_faces_from_directory`, which takes the following arguments:

- `directory_path`: The path to the directory containing the images and videos.
- `output_directory_path`: The path to the directory where the extracted faces should be saved.
- `method`: The face detection method to use. Available options are 'dlib' and 'opencv' (default is 'dlib').

Example usage:

```python
from extract import extract_faces_from_directory

input_directory = "path/to/data/directory"
output_directory = "path/to/output/directory"

extract_faces_from_directory(input_directory, output_directory, method='dlib')

```

## Load
The load process involves creating a CSV file containing the class/person name and the relative path to the extracted faces. This structured format enables easier access to the data and facilitates the subsequent transformation stage.

The main function for the load process is load, which takes the following arguments:

output_directory_path: The path to the directory where the extracted faces are saved.
csv_file_path: The path to the CSV file to create.
Example usage:

```python
from load import load

output_directory = "path/to/output/directory"
csv_file_path = "path/to/csv/file"

load(output_directory, csv_file_path)

```

## Transform

The transformation process is performed during the data augmentation stage. This stage involves applying various data augmentation techniques to the extracted faces to increase the diversity and size of the dataset. Data augmentation can include operations such as rotation, scaling, flipping, and changing brightness/contrast.
