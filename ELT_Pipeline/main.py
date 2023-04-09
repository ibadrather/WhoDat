import os
from extract import extract
from load import load
import sys
import time

# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from Common.logging import log


LOG_FILE_NAME = "etl_logfile.txt"
INPUT_DIR = "/home/ibad/Desktop/WhoDat_Dataset/"
OUTPUT_DIR = "/home/ibad/Desktop/WhoDat_Dataset/WhoDat_Faces/"
METHOD = "dlib"
SIZE = (256, 256)


def main(input_dir: str, output_dir: str, log_file_name: str, method: str, size: tuple) -> None:
    # # 1 - Extract
    # log("Extracting faces from images and videos...", log_file_name)
    # extract(input_dir, output_dir, method, size)
    # log("Finished extracting faces from images and videos.", log_file_name)

    # 2 - Load
    log("Creating CSV file...", log_file_name)
    csv_file_path = os.path.join(output_dir, "all_data.csv")
    load(output_dir, csv_file_path)
    log("Finished creating CSV file.", log_file_name)

    # 3 - Transform
    # Transform for this will be done in the training script.


if __name__ == "__main__":
    os.system("clear")

    start_time = time.time()

    main(INPUT_DIR, OUTPUT_DIR, LOG_FILE_NAME, METHOD, SIZE)

    print("Total time: {} mins".format((time.time() - start_time) / 60))
