
from tensorflow import keras
import sys
sys.path.append('../')
import utils
grid_estimator = keras.models.load_model('./MLP_best_GridSearchCV.h5')
hyperopt_estimator = keras.models.load_model('./MLP_best_TuneSearchCV_hyperopt.h5')
bayesian_estimator = keras.models.load_model('./MLP_best_TuneSearchCV_bayesian.h5')
optuna_estimator = keras.models.load_model('./MLP_best_TuneSearchCV_optuna.h5')

print('Loading dataset')
predictions = ['Animal', 'Animal', 'Animal', 'Human']
train, valid, test = utils.load_learning_data(
    predictions=predictions,
    # root_dir='debug_dataset',
    root_dir='../final_dataset_not_reduced',
    single_dim=True
    )

# dataset = train + valid + test
dataset = valid + test
# dataset.append(train)
# print('Dataset shape', len(dataset), len(dataset[0]))
# dataset.append(valid)
# print('Dataset shape', len(dataset), len(dataset[0]))
# dataset.append(test)
print('Train shape', len(train), len(train[0]))
print('Valid shape', len(valid), len(valid[0]))
print('Test shape', len(test), len(test[0]))
print('Dataset shape', len(dataset), len(dataset[0]))


X_test, Y_test = utils.model_ready_data(dataset)

print('GridSearch evaluation')
loss, acc = grid_estimator.evaluate(
        X_test,
        Y_test
        )

print('TuneSearch with hyperopt evaluation')
loss, acc = hyperopt_estimator.evaluate(
        X_test,
        Y_test
        )

print('TuneSearch with bayesian evaluation')
loss, acc = bayesian_estimator.evaluate(
        X_test,
        Y_test
        )

print('TuneSearch with optuna evaluation')
loss, acc = optuna_estimator.evaluate(
        X_test,
        Y_test
        )
