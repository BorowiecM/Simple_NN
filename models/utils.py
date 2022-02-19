import matplotlib.pyplot as plt
import numpy as np
from pandas import factorize
from tensorflow.keras.utils import to_categorical
from dataset import Dataset
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping

"""
tune_sklearn sometimes causes weird errors.
In this case replace following import with
RandomizedSearchCV and GridSearchCV from sklearn module.
"""
from tune_sklearn import TuneSearchCV, TuneGridSearchCV


def load_learning_data(
    predictions: str,
    root_dir: str = "dataset",
    single_dim: bool = True,
    grayscale: bool = True,
    input_size: int = 32,
):
    ds = Dataset(root_dir, predictions)

    return ds.load_learning_data(
        grayscale=grayscale, input_size=input_size, single_dim=single_dim
    )


def decorated_print(sentence: str = None):
    return print(
        "=========================\n", sentence, "\n========================="
    )


def model_ready_data(
    set: list,
    num_classes: int = 2
    # ) -> tuple[np.array, np.array]:
):
    """
    Returns two DataFrames:
    * model input
    * model predictions
    """
    X = []
    Y = []
    for lst in set:
        X.append(lst[0])
        Y.append(lst[1])

    X = np.asarray(X).astype("float32")
    Y = np.asarray(Y)
    Y = to_categorical(factorize(Y)[0], num_classes=num_classes)
    return X, Y


def get_best_estimator_score_params(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    build_model,
    param_distrib: dict,
    TuneSearch: bool = True,
    search_optimization: str = "bayesian",
    epochs: int = 100,
):
    regressor = KerasRegressor(build_model, verbose=0)
    if TuneSearch:
        search = TuneSearchCV(
            regressor,
            param_distrib,
            verbose=1,
            early_stopping=True,
            max_iters=epochs,
            search_optimization=search_optimization,
        )
    else:
        search = TuneGridSearchCV(
            regressor,
            param_distrib,
            verbose=1,
            early_stopping=True,
            max_iters=epochs,
        )

    search.fit(
        X_train,
        Y_train,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor="accuracy", patience=10)],
        validation_data=(X_valid, Y_valid),
    )

    return (search.best_estimator_, search.best_score_, search.best_params_)


def print_train_val_loss(history, save: bool = True):
    plt.rcParams["figure.figsize"] = (18, 8)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    plt.plot(
        np.arange(1, len(history["loss"]) + 1),
        history["loss"],
        label="Training Loss",
    )
    plt.plot(
        np.arange(1, len(history["loss"]) + 1),
        history["val_loss"],
        label="Validation Loss",
    )
    plt.title("Training vs. Validation Loss", size=20)
    plt.xlabel("Epoch", size=14)
    plt.legend()
    if save:
        plt.savefig("Training vs. Validation Loss.jpg")
    plt.cla()


def print_train_val_acc(history, save: bool = True):
    plt.rcParams["figure.figsize"] = (18, 8)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    plt.plot(
        np.arange(1, len(history["accuracy"]) + 1),
        history["accuracy"],
        label="Training Accuracy",
    )
    plt.plot(
        np.arange(1, len(history["accuracy"]) + 1),
        history["val_accuracy"],
        label="Validation Accuracy",
    )
    plt.title("Training vs. Validation Accuracy", size=20)
    plt.xlabel("Epoch", size=14)
    plt.legend()
    if save:
        plt.savefig("Training vs. Validation Accuracy.jpg")
    plt.cla()
