import matplotlib.pyplot as plt
import numpy as np


def table_plot(columns, data, filename):

    rcolors = plt.cm.YlOrBr(np.full(len(columns), 0.2))

    table = plt.table(
        cellText=data,
        colLabels=columns,
        colColours=rcolors,
        bbox=[0, 0, 1, 1],
        loc="center",
    )
    table.scale(2, 2)
    table.set_fontsize(30)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")


MLP_columns = [
    "Optimizer",
    "Dropout",
    "Units in layer",
    "Layers",
    "Accuracy (%)",
]

MLP_data = [
    ["None", "0.3", "256", "1", "98.88"],
    ["'bayesian'", "0.45", "256", "2", "2.58"],
    ["'hyperopt'", "0.3", "64", "2", "98.11"],
    ["'optuna'", "0.6", "32", "1", "2.23"],
]

table_plot(MLP_columns, MLP_data, "MLP_results.png")


CNN_columns = ["Optimizer", "Dropout", "Layers", "Accuracy (%)"]

CNN_data = [
    ["'bayesian'", "0.3", "1", "99.69"],
    ["'hyperopt'", "0.45", "2", "98.46"],
    ["'optuna'", "0.3", "1", "98.77"],
]

table_plot(CNN_columns, CNN_data, "CNN_results.png")

MLP_data_columns = [
    "Optimizer",
    "Validation accuracy",
    "Test accuracy",
    "Full dataset accuracy",
]

MLP_data_check = [
    ["None", "1.08%", "98.88%", "99.43%"],
    ["'bayesian'", "2.66%", "2.58%", "2.42%"],
    ["'hyperopt'", "2.12%", "98.11%", "98.41%"],
    ["'optuna'", "3.39%", "2.23%", "2.53%"],
]

table_plot(MLP_data_columns, MLP_data_check, "MLP_data_check_results.png")
