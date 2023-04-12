import mlflow
import optuna
import training
import pickle
from argparser import parse_arguments
from optuna import Trial
from optuna.study import Study


def objective(trial: Trial) -> float:
    """
    Optuna objective function that suggests hyperparameters, runs the training, and logs the result to MLflow.

    Args:
        trial (Trial): The Optuna trial object.

    Returns:
        float: The objective function value (validation loss).
    """
    # Suggest hyperparameters
    epochs = trial.suggest_int("epochs", 100, 200, step=30)
    bs = trial.suggest_int("bs", 20, 50)
    lr = trial.suggest_float("lr", 7e-6, 1e-5, step=3e-6)
    arch = trial.suggest_categorical("arch", ["SimpleCNN", "ResNet50", "CNN"])
    sample_size = trial.suggest_int("sample_size", 100, 400, step=100)
    dropout = trial.suggest_float("dropout", 0.1, 0.8, step=0.2)

    # Get the default arguments
    default_args = parse_arguments()

    # Update the default arguments with the suggested hyperparameters
    default_args.epochs = epochs
    default_args.bs = bs
    default_args.lr = lr
    # default_args.patience = patience
    default_args.arch = arch
    default_args.sample_size = sample_size
    default_args.dropout = dropout

    # Run the training
    train_result = training.main(default_args)

    # Log the result to MLflow
    mlflow.log_metric("best_val_loss", train_result)

    return train_result


def save_study(study: Study, trial: Trial, study_name: str) -> None:
    """
    Saves the study to a file after each trial.

    Args:
        study (Study): The Optuna study object.
        trial (Trial): The Optuna trial object.
        study_name (str): The name of the study to save.
    """
    study_file = f"{study_name}.pkl"
    with open(study_file, "wb") as f:
        pickle.dump(study, f)
    print(f"Saved study after trial {trial.number}")


def main():
    study_name = "WhoDat_Model_Dev_optuna_study"
    storage_name = f"sqlite:///{study_name}.db"
    study_file = f"{study_name}.pkl"

    # Check if a saved study exists
    try:
        with open(study_file, "rb") as f:
            study = pickle.load(f)
    except FileNotFoundError:
        # If not, create a new study and save it
        study = optuna.create_study(
            direction="minimize", study_name=study_name, storage=storage_name
        )
        with open(study_file, "wb") as f:
            pickle.dump(study, f)

    # Optimize the study with the objective function and the custom callback
    study.optimize(
        objective,
        n_trials=10,
        callbacks=[lambda study, trial: save_study(study, trial, study_name)],
    )

    # Save the updated study
    with open(study_file, "wb") as f:
        pickle.dump(study, f)

    # Print the best trial
    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ", study.best_trial.params)


if __name__ == "__main__":
    main()
