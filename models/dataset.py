import os
import string
import numpy as np
from pandas import DataFrame, concat
from PIL import Image, ImageOps
from sklearn.utils import shuffle


class Dataset:
    '''
    Dataset object.
    Arguments:
    * directory - relative path to dataset
    * expectations - list containing output classes of categories.
      Categories are subdirectories containing datasets to merge, but belonging
      to different classes.
    '''
    def __init__(self, directory: str, expectations: list):
        self.source_directory = directory
        self.categories = self.get_categories()
        self.expectations = expectations

    '''
    Returns list containing categories from self.source_directory.
    '''
    def get_categories(self) -> list:
        categories = os.listdir(self.source_directory)
        return [category for category in categories if '.' not in category]

    '''
    Returns DataFrame containing all photos from Dataset.
    Also it contains 'expectation' column,
    containing classes from self.expectations.
    Arguments:
    * split_proportions - list containing proportions of train-valid-test sets
    * input_size - defines size of returned images
    * grayscale - defines if image shoul be in grayscale or not
    * single_dim - defines if images should be flattened
    '''
    def load_learning_data(self,
                           split_proportions: list = [0.8, 0.1, 0.1],
                           input_size: int = None,
                           grayscale: bool = False,
                           single_dim: bool = False) -> DataFrame:
        train = DataFrame()
        valid = DataFrame()
        test = DataFrame()

        for index, category in enumerate(self.categories):
            images = self.get_images_from_category(category,
                                                   grayscale,
                                                   input_size,
                                                   single_dim
                                                   )

            images['expectation'] = self.expectations[index]
            images = shuffle(images).reset_index(drop=True)
            size = len(images.index)

            train = concat([
                train,
                images[:int(split_proportions[0] * size)]
                ],
                ignore_index=True
                )

            valid = concat([
                valid,
                images[int(split_proportions[0] * size):int((
                    split_proportions[0] + split_proportions[1]) * size)]
                ],
                ignore_index=True
                )

            test = concat([
                test,
                images[int((
                    split_proportions[0] + split_proportions[1]) * size):]
                ],
                ignore_index=True
                )

        train = shuffle(train).reset_index(drop=True)
        valid = shuffle(valid).reset_index(drop=True)
        test = shuffle(test).reset_index(drop=True)

        return train, valid, test

    '''
    Returns Dataframe containing all images from chosen category.
    Arguments:
    * category - category existing in Dataset
    * grayscale - defines if image should be in grayscale
    * input_size - defines images width and height, in pixels.
    * single_dim - defines if images should be flattened.
    '''
    def get_images_from_category(self,
                                 category: string,
                                 grayscale: bool = False,
                                 input_size: int = None,
                                 single_dim: bool = False) -> DataFrame:
        image_names = os.listdir(self.source_directory + '/' + category)

        images = []
        for image_name in image_names:
            path = self.source_directory + '/' + category + '/' + image_name
            image = self.np_image(path, grayscale, input_size)
            if single_dim:
                image = np.ravel(image) / 255.0
            images.append(image)

        if single_dim:
            return DataFrame(images)
        else:
            '''
            single_dim=False causes error in get_all_images in return,
            i.e. "Must pass 2-d input. shape=(16382, 200, 200)"
            '''
            return DataFrame(images)

    '''
    Returns image as Numpy array, using PIL Image.
    Arguments:
    * path - relative path to file
    * grayscale - defines if image should be in grayscale
    * input_size - defines image width and height, in pixels.
      Undefined means no image scaling.
    '''
    def np_image(self,
                 path: string,
                 grayscale: bool = False,
                 input_size: int = None) -> np.ndarray:
        image = Image.open(path)
        if grayscale:
            image = ImageOps.grayscale(image)
        if input_size:
            image = image.resize(size=(input_size, input_size))

        return np.asarray(image)

    '''
    Returns Dataset prediction classes
    '''
    def num_classes(self) -> int:
        return len(np.unique(self.expectations))
