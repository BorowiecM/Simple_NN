
from tensorflow import keras
import sys
sys.path.append('../')
import utils
hyperopt_estimator = keras.models.load_model('./hyperopt/CNN_best.h5')
bayesian_estimator = keras.models.load_model('./bayesian/CNN_best.h5')
optuna_estimator = keras.models.load_model('./bayesian/CNN_best.h5')

print('Loading dataset')
predictions = ['Animal', 'Animal', 'Animal', 'Human']
train, valid, test = utils.load_learning_data(
    predictions=predictions,
    # root_dir='debug_dataset',
    root_dir='../final_dataset_not_reduced',
    single_dim=False
    )

dataset = train + valid + test

X_test, Y_test = utils.model_ready_data(dataset)

print('Hyperopt evaluation')
loss, acc = hyperopt_estimator.evaluate(
        X_test,
        Y_test
        )

print('Bayesian evaluation')
loss, acc = bayesian_estimator.evaluate(
        X_test,
        Y_test
        )

print('Optuna evaluation')
loss, acc = bayesian_estimator.evaluate(
        X_test,
        Y_test
        )