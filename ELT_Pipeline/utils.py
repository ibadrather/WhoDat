from datetime import datetime


def log(message: str, file_name: str = "logfile.txt") -> None:
    """
    Log a message to a file.
    :param message: Message to log
    :return: None
    """
    # Write your code here
    timestamp_format = "%Y-%m-%d-%H:%M:%S"  # Year-Monthname-Day-Hour-Minute-Second
    now = datetime.now()  # get current timestamp
    timestamp = now.strftime(timestamp_format)
    with open(file_name, "a") as f:
        f.write(timestamp + "," + message + "\n")
