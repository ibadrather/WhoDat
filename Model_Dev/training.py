import sys
import os
from tqdm import tqdm
import pandas as pd
import time
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import required modules
from Common.DataAugmentation import AlbumentationsDataAugmentation
from Common.Trainer import Trainer

import torch
from torch.utils.data import DataLoader

from models import FaceRecognitionModel, SimpleCNN
from dataloading import WhoDatDataset

# Clear console
os.system("clear")

TIMESTAMP = time.strftime("%Y_%m_%d-%H_%M_%S")

# Clear cache and set random seeds for reproducibility
torch.cuda.empty_cache()
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

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
data_augmentation = AlbumentationsDataAugmentation(image_size=(256, 256), options=augmentations)

# Data paths
train_data_path = "/home/ibad/Desktop/WhoDat/WhoDat_Faces/train.csv"
val_data_path = "/home/ibad/Desktop/WhoDat/WhoDat_Faces/val.csv"

# Load the data
train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)

# Initialize the model with the number of classes
num_classes = 9
# model = FaceRecognitionModel(num_classes=num_classes)
model = SimpleCNN(num_classes=num_classes)

# Set up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

# Set up the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Set up the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=5
)

# Set the device to be used for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    output_dir=os.path.join("training_output", TIMESTAMP)
)

# Now let's train the model
# We also have to balance the classes on the fly before every epoch

EPOCHS = 1000
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    # Balance the data by stratified sampling randomly
    balanced_data = train_data.groupby("class").sample(n=100, replace=True, random_state=int(time.time()))

    # Create the balanced dataset
    train_dataset = WhoDatDataset(balanced_data, data_augmentation=data_augmentation)
    val_dataset = WhoDatDataset(val_data, data_augmentation=data_augmentation)

    # Create the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4)

    # Train the model for one epoch
    trainer.train_epoch(train_dataloader)
    trainer.val_epoch(val_dataloader)

    trainer.plot_losses()
