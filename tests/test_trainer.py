import os
import shutil
import sys
import pytest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from Common.Trainer import Trainer


@pytest.fixture(scope="module")
def trainer():
    # Initialize the model, loss function, and optimizer
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=5,
        output_dir=os.path.join("tests", "trainer_test_output"),
    )
    yield trainer

    # Clean up the test output directory after running tests
    shutil.rmtree(trainer.output_dir)


# Create a simple dataset for testing purposes
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=10)
val_loader = DataLoader(dataset, batch_size=10)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc1(x)


def test_training_step(trainer):
    inputs, labels = X[:10], y[:10]
    trainer.training_step(inputs, labels)
    assert trainer.optimizer.state != {}, "Optimizer state not updated"


def test_train_epoch(trainer):
    train_loss = trainer.train_epoch(train_dataloader=train_loader)
    assert isinstance(train_loss, float), "train_epoch should return a float value"


def test_validation_step(trainer):
    inputs, labels = X[:10], y[:10]
    loss = trainer.validation_step(inputs, labels)
    assert isinstance(loss, float), "validation_step should return a float value"


def test_val_epoch(trainer):
    val_loss = trainer.val_epoch(val_dataloader=val_loader)
    assert isinstance(val_loss, float), "val_epoch should return a float value"


def test_model_outputs(trainer):
    inputs, labels = X[:10], y[:10]
    outputs = trainer.model(inputs)
    assert outputs.shape == (
        10,
        2,
    ), "Model outputs should be of shape (batch_size, num_classes)"


def test_train_epochs(trainer):
    trainer.train_epochs(train_dataloader=train_loader, val_dataloader=val_loader)
    assert (
        len(trainer.train_losses) == 5
    ), "Number of recorded train losses should be equal to the number of epochs"
    assert (
        len(trainer.val_losses) == 5
    ), "Number of recorded val losses should be equal to the number of epochs"


def test_save_checkpoint(trainer):
    trainer.save_checkpoint(epoch=1)
    assert os.path.exists(
        os.path.join(trainer.output_dir, "{}.pth".format(trainer.architecture))
    ), "Model checkpoint should be saved as a .pth file"
    assert os.path.exists(
        os.path.join(trainer.output_dir, "{}.pt".format(trainer.architecture))
    ), "Model checkpoint should be saved as a .pt file (TorchScript)"


def test_plot_losses(trainer):
    trainer.plot_losses()
    assert os.path.isfile(
        os.path.join(trainer.output_dir, "losses.png")
    ), "Losses plot should be saved as an image file"
