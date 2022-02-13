from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


class MLP:
    def __init__(self):
        self.single_dim = True
        self.param_distrib = {
            "model__hidden_units": [16, 32, 64, 128, 256, 512],
            "model__dropout": [0.3, 0.45, 0.6, 0.7],
            "model__layers": [1, 2, 3, 4, 5]
        }
        self.build_model = build_model


def build_model(
        hidden_units: int = 256,
        dropout: float = 0.45,
        layers: int = 1
        ) -> Sequential:
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
