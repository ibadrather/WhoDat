import mlflow
import optuna
import training
from argparser import parse_arguments


def objective(trial):
    # Suggest hyperparameters
    epochs = trial.suggest_int("epochs", 100, 200)
    bs = trial.suggest_int("bs", 20, 50)
    lr = trial.suggest_loguniform("lr", 1e-6, 2e-5)
    # patience = trial.suggest_int("patience", 10,)
    arch = trial.suggest_categorical("arch", ["SimpleCNN", "ResNet50"])
    sample_size = trial.suggest_int("sample_size", 100, 300)

    # Get the default arguments
    default_args = parse_arguments()

    # Update the default arguments with the suggested hyperparameters
    default_args.epochs = epochs
    default_args.bs = bs
    default_args.lr = lr
    # default_args.patience = patience
    default_args.arch = arch
    default_args.sample_size = sample_size

    # Run the training
    train_result = training.main(default_args)

    # Log the run with MLflow
    with mlflow.start_run():
        # Log arguments to MLflow
        for arg, value in vars(default_args).items():
            mlflow.log_param(arg, value)

        # Log the result to MLflow
        mlflow.log_metric("best_val_loss", train_result)

    return train_result


def main():
    # Set up MLflow experiment
    mlflow.set_experiment("facial_recognition_training_optuna")

    # Create the Optuna study
    study = optuna.create_study(direction="minimize")

    # Optimize the study with the objective function
    study.optimize(objective, n_trials=10)

    # Print the best trial
    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ", study.best_trial.params)


if __name__ == "__main__":
    main()
