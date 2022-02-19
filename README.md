# Simple_NN

Simple neural network, which can tell whether the face belongs to a human or an animal

## Download links

To download initial dataset and result archives, use the following link: <https://uniwersytetlodzki-my.sharepoint.com/:f:/g/personal/ul0237204_edu_uni_lodz_pl/EmD0Q5g44hpDum-2Nz4NCnYBr6jOS2xevm857FXMQWoU6Q?e=7JlUbK>

## Dataset processing

To perform dataset processing, unpack `datasets.zip` archive from link above into `dataset_processing` folder.
Then run file `main.py` from this folder.
When run with argument `--crop`, it indicates that images should be cropped.
Already processed files are in two archives: `results.zip` and `crop_results`.
A `crop_results` archive contains cropped faces from all datasets, and `results` archive contains source images with red bounding boxes around faces.

## Model training

Before running training, please unpack `final_dataset_not_reduced.zip` archive into `models` folder.
To run model training please run `main.py` file from the same directory.
To debug code, please unpack `debug_dataset.zip` archive into `models` folder, and add appropriate flags in `main` method in `main.py` file.
It causes to load a smaller dataset and lowers training epochs, so whole script runs faster, and is easier to debug.
