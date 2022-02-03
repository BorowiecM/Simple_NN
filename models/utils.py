from pandas import DataFrame
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

'''
Returns two DataFrames:
* model input
* model predictions
'''


def model_ready_data(set: DataFrame,
                     num_classes: int = 2) -> tuple[DataFrame, DataFrame]:
    X = set.drop('expectation', axis=1)
    Y = set['expectation']
    Y = to_categorical(Y.factorize()[0], num_classes=num_classes)
    return X, Y


'''
hidden layers:
* layers count - 1-5 layers, depending on problem
* neuron count - 10-100
* activation function - ReLU or SELU
output layer:
* output neurons - 1 for each class
* activation function - softmax
loss: crossentropy
'''


def build_model(hidden_units=256, dropout=0.45, layers=1):
    model = Sequential([
        Dense(hidden_units, input_dim=32**2, activation='relu'),
        Dropout(dropout)
    ])
    for _ in range(layers):
        model.add(Dense(hidden_units, activation='relu'))
        model.add(Dropout(dropout))

    model.add(Dense(2, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics='accuracy'
    )

    return model


def build_keras_tuner_model(hp):
    hidden_units = hp.Int('hidden_units', min_value=32, max_value=512, step=32)
    dropout = hp.Float('dropout', min_value=0.3, max_value=0.75, step=0.15)
    layers = hp.Int('layers', min_value=1, max_value=5, step=1)
    model = build_model(
        hidden_units,
        dropout,
        layers
    )
    return model
