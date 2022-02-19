import os
import random
import numpy as np
from PIL import Image, ImageOps


class Dataset:
    def __init__(self, directory: str, expectations: list):
        """
        Dataset object.
        Arguments:
        * directory - relative path to dataset
        * expectations - list containing output classes of categories.
        Categories are subdirectories containing datasets to merge,
        but belonging to different classes.
        """
        self.source_directory = directory
        self.categories = self.get_categories()
        self.expectations = expectations

    def get_categories(self) -> list:
        """
        Returns list containing categories from self.source_directory.
        """
        categories = os.listdir(self.source_directory)
        return [category for category in categories if "." not in category]

    def load_learning_data(
        self,
        split_proportions: list = [0.8, 0.1, 0.1],
        input_size: int = None,
        grayscale: bool = False,
        single_dim: bool = False,
    ) -> list:
        """
        Returns DataFrame containing all photos from Dataset.
        Also it contains 'expectation' column,
        containing classes from self.expectations.
        Arguments:
        * split_proportions - list with proportions of train-valid-test sets
        * input_size - defines size of returned images
        * grayscale - defines if image shoul be in grayscale or not
        * single_dim - defines if images should be flattened
        """
        train = []
        valid = []
        test = []

        for index, category in enumerate(self.categories):
            images = self.get_images_from_category(
                category, grayscale, input_size, single_dim
            )

            for lst in images:
                lst.append(self.expectations[index])

            random.shuffle(images)
            size = len(images)

            train.extend(images[: int(split_proportions[0] * size)])
            valid.extend(
                images[
                    int(split_proportions[0] * size) : int(
                        (split_proportions[0] + split_proportions[1]) * size
                    )
                ]
            )
            test.extend(
                images[
                    int((split_proportions[0] + split_proportions[1]) * size) :
                ]
            )

        random.shuffle(train)
        random.shuffle(valid)
        random.shuffle(test)

        return train, valid, test

    def get_images_from_category(
        self,
        category: str,
        grayscale: bool = False,
        input_size: int = None,
        single_dim: bool = False,
    ) -> np.array:
        """
        Returns Dataframe containing all images from chosen category.
        Arguments:
        * category - category existing in Dataset
        * grayscale - defines if image should be in grayscale
        * input_size - defines images width and height, in pixels.
        * single_dim - defines if images should be flattened.
        """
        image_names = os.listdir(self.source_directory + "/" + category)

        images = []
        for image_name in image_names:
            path = self.source_directory + "/" + category + "/" + image_name
            image = self.np_image(path, grayscale, input_size)
            image = image.astype("float32")
            if single_dim:
                image = np.ravel(image)
            image /= 255.0
            images.append([image])

        if single_dim:
            return images
        else:
            """
            single_dim=False causes error in get_all_images in return,
            i.e. "Must pass 2-d input. shape=(16382, 200, 200)"
            """
            return images

    def np_image(
        self, path: str, grayscale: bool = False, input_size: int = None
    ) -> np.array:
        """
        Returns image as Numpy array, using PIL Image.\n
        Arguments:
        * path - relative path to file
        * grayscale - defines if image should be in grayscale
        * input_size - defines image width and height, in pixels.
        Undefined means no image scaling.
        """
        image = Image.open(path)
        if grayscale:
            image = ImageOps.grayscale(image)
        if input_size:
            image = image.resize(size=(input_size, input_size))

        return np.asarray(image)

    def num_classes(self) -> int:
        """
        Returns Dataset prediction classes
        """
        return len(np.unique(self.expectations))
