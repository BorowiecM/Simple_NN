from mlp import MLP
from cnn import CNN
import utils
from PIL import Image
import numpy as np


if __name__ == '__main__':
    arch = CNN()
    # arch = MLP()

    utils.decorated_print('Loading dataset')
    predictions = ['Animal', 'Animal', 'Animal', 'Human']
    train, valid, test = utils.load_learning_data(
        predictions=predictions,
        # root_dir='debug_dataset',
        root_dir='final_dataset_not_reduced',
        single_dim=arch.single_dim
        )

    X_train, Y_train = utils.model_ready_data(train)
    X_valid, Y_valid = utils.model_ready_data(valid)
    X_test, Y_test = utils.model_ready_data(test)

    utils.decorated_print('Finding best estimator')
    (best_estimator,
     best_score,
     best_params) = utils.get_best_estimator_score_params(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        arch.build_model,
        arch.param_distrib,
        # TuneSearch = False,
        # epochs=5
        epochs=100
        )

    utils.decorated_print('Saving model and its learning curves')
    utils.print_train_val_loss(best_estimator.history_)
    utils.print_train_val_acc(best_estimator.history_)
    best_estimator.model_.save("CNN_best.h5")

    utils.decorated_print('Best results:')
    print('- params:', best_params)
    print('- score:', best_score)

    utils.decorated_print('Evaluating best model')
    loss, acc = best_estimator.model_.evaluate(
        X_test,
        Y_test
        )
    utils.decorated_print('Evaluation results:')
    print('- loss:', loss)
    print('- accuracy:', acc)

    utils.decorated_print('Predicting test images')
    pred = best_estimator.model_.predict(
        X_test
    )
    print('Predicted type for first test image:',
          'Animal' if pred[0][0] > pred[0][1] else 'Human')
    test_image = X_test[0]
    input_image = Image.fromarray(np.uint8(test_image * 255).reshape((32, 32)))
    input_image.save('sample_image.jpg')
