import argparse

import mlflow
import optuna
import training


def objective(trial):
    # Suggest hyperparameters
    epochs = trial.suggest_int("epochs", 500, 300, 1000)
    bs = trial.suggest_int("bs", 30, 50)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-5, 1e-6)
    patience = trial.suggest_int("patience", 15)
    arch = trial.suggest_categorical(
        "arch", ["SimpleCNN", "FaceRecognitionModel", "ResNet18"]
    )

    # Create an argparse.Namespace object with the suggested hyperparameters
    arg_namespace = argparse.Namespace(
        epochs=epochs,
        bs=bs,
        lr=lr,
        patience=patience,
        arch=arch,
        output_dir="training_output",
        seed=42,
        device="cuda:0",
        optimizer="Adam",
        scheduler="ReduceLROnPlateau",
        loss_fn="CrossEntropyLoss",
        image_size=(224, 224),
    )

    # Run the training
    train_result = training.main(arg_namespace)

    # Log the run with MLflow
    with mlflow.start_run():
        # Log arguments to MLflow
        for arg, value in vars(arg_namespace).items():
            mlflow.log_param(arg, value)

    return train_result


def main():
    # Set up MLflow experiment
    mlflow.set_experiment("facial_recognition_training_optuna")

    # Create the Optuna study
    study = optuna.create_study(direction="maximize")

    # Optimize the study with the objective function
    study.optimize(objective, n_trials=10)

    # Print the best trial
    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ", study.best_trial.params)


if __name__ == "__main__":
    main()
