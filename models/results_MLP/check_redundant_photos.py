from tensorflow import keras
import numpy as np
import sys

sys.path.append("../")
import utils

grid_estimator = keras.models.load_model("./MLP_best_GridSearchCV.h5")
hyperopt_estimator = keras.models.load_model(
    "./MLP_best_TuneSearchCV_hyperopt.h5"
)
bayesian_estimator = keras.models.load_model(
    "./MLP_best_TuneSearchCV_bayesian.h5"
)
optuna_estimator = keras.models.load_model("./MLP_best_TuneSearchCV_optuna.h5")

print("Loading dataset")
predictions = ["Animal", "Animal", "Animal", "Human"]
train, valid, test = utils.load_learning_data(
    predictions=predictions,
    # root_dir='debug_dataset',
    root_dir="../final_dataset_not_reduced",
    single_dim=True,
)

dataset = train + valid + test
# dataset = valid + test

X_test, Y_test = utils.model_ready_data(dataset)

for num in range(5):
    print("GridSearch evaluation")
    pred = grid_estimator.predict(X_test)
    indices = []
    for i in range(len(pred)):
        if pred[i][0] > pred[i][1]:
            pred[i][0] = 0.0
            pred[i][1] = 1.0
        else:
            pred[i][0] = 1.0
            pred[i][1] = 0.0
        if pred[i][0] != Y_test[i][0]:
            indices.append(i)
    subset_of_wrongly_predicted = [X_test[i] for i in indices]
    np.save(
        str(num) + "resultsgs.txt",
        subset_of_wrongly_predicted,
        allow_pickle=True,
    )
    print("Bad predictions:", len(subset_of_wrongly_predicted))

    print("TuneSearch with hyperopt evaluation")
    pred = hyperopt_estimator.predict(X_test)
    indices = []
    for i in range(len(pred)):
        if pred[i][0] > pred[i][1]:
            pred[i][0] = 0.0
            pred[i][1] = 1.0
        else:
            pred[i][0] = 1.0
            pred[i][1] = 0.0
        if pred[i][0] != Y_test[i][0]:
            indices.append(i)
    subset_of_wrongly_predicted = [X_test[i] for i in indices]
    np.save(
        str(num) + "resultsho.txt",
        subset_of_wrongly_predicted,
        allow_pickle=True,
    )
    print("Bad predictions:", len(subset_of_wrongly_predicted))

    print("TuneSearch with bayesian evaluation")
    pred = bayesian_estimator.predict(X_test)
    indices = []
    for i in range(len(pred)):
        if pred[i][0] > pred[i][1]:
            pred[i][0] = 0.0
            pred[i][1] = 1.0
        else:
            pred[i][0] = 1.0
            pred[i][1] = 0.0
        if pred[i][0] != Y_test[i][0]:
            indices.append(i)
    subset_of_wrongly_predicted = [X_test[i] for i in indices]
    np.save(
        str(num) + "resultsbs.txt",
        subset_of_wrongly_predicted,
        allow_pickle=True,
    )
    print("Bad predictions:", len(subset_of_wrongly_predicted))

    print("TuneSearch with optuna evaluation")
    pred = optuna_estimator.predict(X_test)
    indices = []
    for i in range(len(pred)):
        if pred[i][0] > pred[i][1]:
            pred[i][0] = 0.0
            pred[i][1] = 1.0
        else:
            pred[i][0] = 1.0
            pred[i][1] = 0.0
        if pred[i][0] != Y_test[i][0]:
            indices.append(i)
    subset_of_wrongly_predicted = [X_test[i] for i in indices]
    np.save(
        str(num) + "resultsot.txt",
        subset_of_wrongly_predicted,
        allow_pickle=True,
    )
    print("Bad predictions:", len(subset_of_wrongly_predicted))
