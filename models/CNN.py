from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import utils
from PIL import Image
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
# To use TuneSearchCV, script must be launched on Linux system or WSL
from tune_sklearn import TuneSearchCV
from dataset import Dataset


def build_model(
        dropout: float = 0.5,
        conv_layers: int = 2,
        grayscale: bool = True
        ):
    '''
    strojenie:
    - ilość trójki warstw
    - dropout
    '''
    model = Sequential([
        layers.Conv2D(64, 7, activation='relu', padding='same',
                      input_shape=[32, 32, 1 if grayscale else 3]),
        layers.MaxPooling2D(2)
        ])

    multiplier = 1

    for _ in range(conv_layers):
        multiplier *= 2
        model.add(layers.Conv2D(
            64 * multiplier, 3, activation='relu', padding='same'))
        model.add(layers.Conv2D(
            64 * multiplier, 3, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(2))

    model.add(layers.Flatten())

    while multiplier > 1:
        multiplier /= 2
        model.add(layers.Dense(64 * multiplier, activation='relu'))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(2, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics='accuracy'
    )

    return model


def load_learning_data(input_size: int, grayscale: bool = True):
    ds = Dataset('final_dataset_not_reduced',
    # ds = Dataset('debug_dataset',
                 ['Animal', 'Animal', 'Animal', 'Human'])
    return ds.load_learning_data(grayscale=grayscale,
                                 input_size=input_size,
                                 single_dim=False)


def get_best_model_score_params(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        param_distrib: dict,
        TuneSearch: bool = True,
        search_optimization: str = 'bayesian',
        epochs: int = 100
        ):
    regressor = KerasRegressor(build_model)
    if TuneSearch:
        search = TuneSearchCV(
            regressor,
            param_distrib,
            search_optimization=search_optimization
            )
    else:
        search = GridSearchCV(
            regressor,
            param_distrib
            )

    search.fit(
        X_train,
        Y_train,
        epochs=epochs,
        validation_data=(X_valid, Y_valid),
        callbacks=[EarlyStopping(monitor='accuracy', patience=10)]
    )

    return (
        search.best_estimator_.model,
        search.best_score_,
        search.best_params_
    )


if __name__ == '__main__':
    print('Loading dataset')
    train, valid, test = load_learning_data(input_size=32)
    X_train, Y_train = utils.model_ready_data(train)
    X_valid, Y_valid = utils.model_ready_data(valid)
    X_test, Y_test = utils.model_ready_data(test)

    MLP_param_distrib = {
        "dropout": [0.3, 0.45, 0.6, 0.7],
        # "conv_layers": [1, 2, 3, 4, 5]
        "conv_layers": [1, 2, 3, 4]
    }

    print('Finding best model')
    best_model, best_score, best_params = get_best_model_score_params(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        MLP_param_distrib,
        # TuneSearch=False,
        # epochs=5
        epochs=100
        )

    # Save model and its learning plots
    utils.print_train_val_loss(best_model.history)
    utils.print_train_val_acc(best_model.history)
    best_model.save("CNN_best.h5")

    print()
    print('Best results:')
    print('- params:', best_params)
    print('- score:', best_score)

    # Evaluate best model
    loss, acc = best_model.evaluate(
        X_test,
        Y_test
        )
    print()
    print('Evaluation results:')
    print('- loss:', loss)
    print('- accuracy:', acc)

    # Predict sample image
    pred = best_model.predict(
        X_test
    )
    print()
    print('Predicted type:', 'Animal' if pred[0][0] > pred[0][1] else 'Human')
    test_image = X_test[0]
    input_image = Image.fromarray(np.uint8(test_image * 255).reshape((32, 32)))
    input_image.show()
    input_image.save('CNN_sample_image.jpg')
