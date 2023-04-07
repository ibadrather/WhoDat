import os
from Common.logging import log
from extract import extract
from load import load

LOG_FILE_NAME = "etl_logfile.txt"
INPUT_DIR = "/media/ibad/7A28-1119/WhoDat_Dataset/"
OUTPUT_DIR = "/media/ibad/7A28-1119/WhoDat_Dataset/WhoDat_Faces/"
METHOD = "dlib"


def main(input_dir: str, output_dir: str, log_file_name: str, method: str) -> None:
    # 1 - Extract
    log("Extracting faces from images and videos...", log_file_name)
    extract(input_dir, output_dir, method)
    log("Finished extracting faces from images and videos.", log_file_name)

    # 2 - Load
    log("Creating CSV file...", log_file_name)
    csv_file_path = os.path.join(output_dir, "faces.csv")
    load(output_dir, csv_file_path)
    log("Finished creating CSV file.", log_file_name)

    # 3 - Transform
    # Transform for this will be done in the training script.


if __name__ == "__main__":
    os.system("clear")
    main(INPUT_DIR, OUTPUT_DIR, LOG_FILE_NAME, METHOD)
