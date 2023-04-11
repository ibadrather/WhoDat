import sys
import os
from tqdm import tqdm
import pandas as pd
import time
from argparser import parse_arguments
from utils import log_arguments_to_mlflow

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import required modules
from Common.DataAugmentation import AlbumentationsDataAugmentation
from Common.Trainer import Trainer

import torch
from torch.utils.data import DataLoader

from models import FaceRecognitionModel, SimpleCNN
from dataloading import WhoDatDataset

from tensorboardX import SummaryWriter
import mlflow.pytorch


def main(arg_namespace=None):
    # Clear console
    os.system("clear")

    # Parse arguments
    if arg_namespace is None:
        args = parse_arguments()
    else:
        args = arg_namespace

    TIMESTAMP = time.strftime("%Y_%m_%d-%H_%M_%S")

    # Clear cache and set random seeds for reproducibility
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Data augmentation
    augmentations = {
        "horizontal_flip": True,
        "vertical_flip": True,
        "random_rotate_90": True,
        "transpose": True,
        "medium_augmentations": True,
        "clahe": True,
        "random_brightness_contrast": True,
        "random_gamma": True,
    }
    data_augmentation = AlbumentationsDataAugmentation(
        image_size=args.image_size, options=augmentations
    )

    # Data paths
    train_data_path = "/home/ibad/Desktop/WhoDat/WhoDat_Faces/train.csv"
    val_data_path = "/home/ibad/Desktop/WhoDat/WhoDat_Faces/val.csv"

    # Load the data
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)

    # Initialize the model with the number of classes
    num_classes = 9
    if args.arch == "FaceRecognitionModel":
        model = FaceRecognitionModel(num_classes=num_classes)
    elif args.arch == "SimpleCNN":
        model = SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError("Invalid model architecture.")

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Set up the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set up the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # Set the device to be used for training
    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )

    # Initialize the Trainer with required parameters
    trainer = Trainer(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        architecture="FaceRecognitionModel",
        device=device,
        patience=20,
        output_dir=os.path.join("training_output", TIMESTAMP),
    )

    # Now let's train the model
    # We also have to balance the classes on the fly before every epoch

    EPOCHS = args.epochs

    mlflow.set_experiment("WhoDat_Experiment")
    with mlflow.start_run():
        # Log the arguments to MLFlow
        log_arguments_to_mlflow(args)

        for epoch in tqdm(range(EPOCHS), desc="Epochs"):
            # Balance the data by stratified sampling randomly
            balanced_data = train_data.groupby("class").sample(
                n=args.sample_size, replace=True, random_state=int(time.time())
            )

            # Create the balanced dataset
            train_dataset = WhoDatDataset(
                balanced_data, data_augmentation=data_augmentation
            )
            val_dataset = WhoDatDataset(val_data, data_augmentation=data_augmentation)

            # Create the DataLoaders
            train_dataloader = DataLoader(
                train_dataset, batch_size=args.bs, shuffle=True
            )
            val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

            # Train the model for one epoch
            trainer.train_epoch(train_dataloader)
            trainer.val_epoch(val_dataloader)

            # Log the training and validation losses to MLflow
            mlflow.log_metric("train_loss", trainer.train_losses[-1], step=epoch)
            mlflow.log_metric("val_loss", trainer.val_losses[-1], step=epoch)

            trainer.plot_losses()
            # plot artifact
            mlflow.log_artifact(
                os.path.join(trainer.output_dir, "losses.png"), "losses.png"
            )

        # Log the final PyTorch model to MLflow
        mlflow.pytorch.log_model(model, "model")
