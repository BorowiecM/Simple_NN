import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Sequential


class CNN:
    def __init__(self):
        self.single_dim = False
        self.param_distrib = {
            "model__dropout": [0.3, 0.45, 0.6, 0.7],
            # "model__conv_layers": [1, 2, 3, 4, 5]
            "model__conv_layers": [1, 2, 3, 4]
        }
        self.build_model = build_model


def build_model(
        dropout: float = 0.5,
        conv_layers: int = 2,
        grayscale: bool = True
        ):
    '''
    tuning:
    - count of tri-layer modules
    - dropout
    '''
    model = Sequential([
        Layers.Conv2D(64, 7, activation='relu', padding='same',
                      input_shape=[32, 32, 1 if grayscale else 3]),
        Layers.MaxPooling2D(2)
        ])

    multiplier = 1

    for _ in range(conv_layers):
        multiplier *= 2
        model.add(Layers.Conv2D(
            64 * multiplier, 3, activation='relu', padding='same'))
        model.add(Layers.Conv2D(
            64 * multiplier, 3, activation='relu', padding='same'))
        model.add(Layers.MaxPooling2D(2))

    model.add(Layers.Flatten())

    # while multiplier > 1:
    #     multiplier /= 2
    #     model.add(layers.Dense(64 * multiplier, activation='relu'))
    #     model.add(layers.Dropout(dropout))

    model.add(Layers.Dense(64, activation='relu'))
    model.add(Layers.Dropout(dropout))
    model.add(Layers.Dropout(dropout))
    model.add(Layers.Dense(2, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics='accuracy'
    )

    return model
