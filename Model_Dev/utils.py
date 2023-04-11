import mlflow
import argparse


def log_arguments_to_mlflow(args: argparse.Namespace) -> None:
    """
    Log the command-line arguments to MLflow.

    Args:
        args: argparse.Namespace object containing the parsed command-line arguments
    """
    for arg, value in vars(args).items():
        mlflow.log_param(arg, value)
