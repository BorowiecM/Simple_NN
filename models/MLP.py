import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import GridSearchCV
# To use TuneSearchCV, script must be launched on Linux system or WSL
from tune_sklearn import TuneSearchCV
from dataset import Dataset
from utils import model_ready_data, build_model


print('Loading dataset')
# load dataset
input_size = 32
ds = Dataset('final_dataset_not_reduced',
             ['Animal', 'Animal', 'Animal', 'Human'])
train, valid, test = ds.load_learning_data(grayscale=True,
                                           input_size=input_size,
                                           single_dim=True)

print('Loading model ready sets')
# load model ready sets
X_train, Y_train = model_ready_data(train)
X_valid, Y_valid = model_ready_data(valid)
X_test, Y_test = model_ready_data(test)

# create GridSearchCV
keras_reg = KerasRegressor(build_model)
param_distrib = {
    "hidden_units": [16, 32, 64, 128, 256, 512],
    "dropout": [0.3, 0.45, 0.6, 0.7],
    "layers": [1, 2, 3, 4, 5]
}
# grid_search = GridSearchCV(keras_reg, param_distrib)
tune_search = TuneSearchCV(
    keras_reg,
    param_distrib,
    search_optimization='bayesian'
    )

# print('Processing GridSearchCV')
print('Processing TuneSearchCV')
# Process GridSearchCV
epochs = 5
# history = grid_search.fit(
tune_search.fit(
    X_train,
    Y_train,
    epochs=epochs,
    validation_data=(X_valid, Y_valid),
    callbacks=[EarlyStopping(monitor='accuracy', patience=10)]
)

# print('GridSearchCV best results')
# Print best results and save model
# print(grid_search.best_params_)
# print(grid_search.best_score_)
# best_model = grid_search.best_estimator_.model
print('TuneSearchCV best results')
print(tune_search.best_params_)
print(tune_search.best_score_)
best_model = tune_search.best_estimator_.model
history = best_model.history
best_model.save("MLP_best.h5")

# Evaluate best model
print('Evaluation')
loss, acc = best_model.evaluate(
    X_test,
    Y_test
    )

print('Sample prediction')
# Predict sample image
pred = best_model.predict(
    X_test
)
print('Prediction:', 'Animal' if pred[0][0] > pred[0][1] else 'Human')
test_image = pd.DataFrame(X_test.loc[0, :])
Image.fromarray(np.uint8(test_image * 255).reshape((32, 32))).show()

# Draw results
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
plt.savefig('Training vs. Validation Loss.jpg')
plt.cla()

plt.plot(np.arange(1, len(history.history['accuracy']) + 1),
         history.history['accuracy'],
         label='Training Accuracy')
plt.plot(np.arange(1, len(history.history['accuracy']) + 1),
         history.history['val_accuracy'],
         label='Validation Accuracy')
plt.title('Training vs. Validation Accuracy', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()
plt.savefig('Training vs. Validation Accuracy.jpg')
plt.cla()
