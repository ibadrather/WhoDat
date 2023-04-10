import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def make_csv_from_dataset_directory(
    dataset_directory_path: str, csv_file_path: str
) -> None:
    """
    Creates a CSV file containing the class/person name and the relative path to the extracted faces.

    Args:
        dataset_directory_path (str): The path to the directory where the extracted faces are saved.
        csv_file_path (str): The path to the CSV file to create.
    """

    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["class", "file_path"])

        for folder_name in os.listdir(dataset_directory_path):
            folder_path = os.path.join(dataset_directory_path, folder_name)

            # Check if the path is a directory
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    # Check if the path is a file
                    if os.path.isfile(file_path):
                        # Write the class and file path to the CSV file
                        relative_path = os.path.join(
                            "WhoDat_Faces", folder_name, file_name
                        )
                        csv_writer.writerow([folder_name, relative_path])


def load(output_directory_path: str, csv_file_path: str) -> None:
    """
    Creates a CSV file containing the class/person name and the relative path to the extracted faces.
    Splits the data into training, validation, and test sets.
    Saves the dataframes to CSV files.

    Args:
        output_directory_path (str): The path to the directory where the extracted faces are saved.
        csv_file_path (str): The path to the CSV file to create.

    Returns:
        None
    """

    # Create the CSV file
    make_csv_from_dataset_directory(output_directory_path, csv_file_path)

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Split the data into training, validation, and test sets
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["class"]
    )

    test_df, val_df = train_test_split(
        val_df, test_size=0.5, random_state=42, stratify=val_df["class"]
    )

    # Save the dataframes to CSV files
    train_df.to_csv(os.path.join(output_directory_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_directory_path, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_directory_path, "test.csv"), index=False)

    # print dimensions of each set
    print("Training set dimensions: ", train_df.shape)
    print("Validation set dimensions: ", val_df.shape)
    print("Test set dimensions: ", test_df.shape)

    return
