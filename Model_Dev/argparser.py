import argparse


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for training a facial recognition model.

    Returns:
        argparse.Namespace: An object containing parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Train a facial recognition model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--bs", type=int, default=40, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--patience", type=int, default=3, help="Patience for early stopping."
    )
    parser.add_argument(
        "--arch", type=str, default="ResNet50", help="Model architecture."
    )
    parser.add_argument(
        "--output_dir", type=str, default="training_output", help="Output directory."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training."
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Optimizer to use for training."
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ReduceLROnPlateau",
        help="Scheduler to use for training.",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="CrossEntropyLoss",
        help="Loss function to use for training.",
    )
    parser.add_argument(
        "--image_size", nargs=2, type=int, default=(224, 224), help="Image size."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Sample size for stratified sampling.",
    )
    return parser.parse_known_args()[0]


# def parse_arguments() -> argparse.Namespace:
#     """
#     Parse command-line arguments for training a facial recognition model.

#     Returns:
#         argparse.Namespace: An object containing parsed command-line arguments
#     """
#     parser = argparse.ArgumentParser(description="Train a facial recognition model.")

#     parser.add_argument("--epochs", type=int, default=300, help="Number of epochs.")
#     parser.add_argument("--bs", type=int, default=40, help="Batch size.")
#     parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
#     parser.add_argument(
#         "--patience", type=int, default=20, help="Patience for early stopping."
#     )
#     parser.add_argument(
#         "--arch", type=str, default="SimpleCNN", help="Model architecture."
#     )
#     parser.add_argument(
#         "--output_dir", type=str, default="training_output", help="Output directory."
#     )

#     parser.add_argument("--seed", type=int, default=42, help="Random seed.")
#     parser.add_argument(
#         "--device", type=str, default="cuda:0", help="Device to use for training."
#     )

#     # optimizer, scheduler, loss_fn
#     parser.add_argument(
#         "--optimizer", type=str, default="Adam", help="Optimizer to use for training."
#     )
#     parser.add_argument(
#         "--scheduler",
#         type=str,
#         default="ReduceLROnPlateau",
#         help="Scheduler to use for training.",
#     )
#     parser.add_argument(
#         "--loss_fn",
#         type=str,
#         default="CrossEntropyLoss",
#         help="Loss function to use for training.",
#     )

#     # image_size
#     parser.add_argument(
#         "--image_size", nargs=2, type=int, default=(224, 224), help="Image size."
#     )

#     # sample size
#     parser.add_argument(
#         "--sample_size",
#         type=int,
#         default=300,
#         help="Sample size for stratified sampling.",
#     )

#     args = parser.parse_args()
#     return args
