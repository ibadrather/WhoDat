import os
import csv


def load(output_directory_path: str, csv_file_path: str) -> None:
    """
    Creates a CSV file containing the class/person name and the relative path to the extracted faces.

    Args:
        output_directory_path (str): The path to the directory where the extracted faces are saved.
        csv_file_path (str): The path to the CSV file to create.
    """

    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["class", "file_path"])

        for folder_name in os.listdir(output_directory_path):
            folder_path = os.path.join(output_directory_path, folder_name)

            # Check if the path is a directory
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    # Check if the path is a file
                    if os.path.isfile(file_path):
                        # Write the class and file path to the CSV file
                        relative_path = os.path.join(
                            "WhosDat_Faces", folder_name, file_name
                        )
                        csv_writer.writerow([folder_name, relative_path])
