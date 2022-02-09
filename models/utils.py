import matplotlib.pyplot as plt
import numpy as np
from pandas import factorize
from tensorflow.keras.utils import to_categorical


def model_ready_data(
        set: list,
        num_classes: int = 2
        ) -> tuple[np.array, np.array]:
    '''
    Returns two DataFrames:
    * model input
    * model predictions
    '''
    # X = np.empty((0, *set[0][0].shape))
    # Y = np.empty((0, 1))
    X = []
    Y = []
    for lst in set:
        # X = np.append(X, lst[0], axis=0)
        # Y = np.append(Y, lst[1])
        X.append(lst[0])
        Y.append(lst[1])

    X = np.asarray(X).astype('float32')
    Y = np.asarray(Y)
    Y = to_categorical(factorize(Y)[0], num_classes=num_classes)
    return X, Y


def print_train_val_loss(history, save: bool = True):
    plt.rcParams['figure.figsize'] = (18, 8)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    plt.plot(np.arange(1, len(history.history['loss']) + 1),
             history.history['loss'],
             label='Training Loss')
    plt.plot(np.arange(1, len(history.history['loss']) + 1),
             history.history['val_loss'],
             label='Validation Loss')
    plt.title('Training vs. Validation Loss', size=20)
    plt.xlabel('Epoch', size=14)
    plt.legend()
    if save:
        plt.savefig('Training vs. Validation Loss.jpg')
    plt.cla()


def print_train_val_acc(history, save: bool = True):
    plt.rcParams['figure.figsize'] = (18, 8)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    plt.plot(np.arange(1, len(history.history['accuracy']) + 1),
             history.history['accuracy'],
             label='Training Accuracy')
    plt.plot(np.arange(1, len(history.history['accuracy']) + 1),
             history.history['val_accuracy'],
             label='Validation Accuracy')
    plt.title('Training vs. Validation Accuracy', size=20)
    plt.xlabel('Epoch', size=14)
    plt.legend()
    if save:
        plt.savefig('Training vs. Validation Accuracy.jpg')
    plt.cla()
