import argparse
from mtcnn.mtcnn import MTCNN
import os
from PIL import Image, ImageDraw
import numpy as np

from queue import Queue
from threading import Thread


class DatasetProcessor:
    def __init__(self, file_list, crop):
        self.crop = crop

        self.queue = Queue()
        for item in file_list:
            self.queue.put(item)

        self.threads = []
        for _ in range(os.cpu_count()):
            self.threads.append(Thread(target=self.process))

    def start(self):
        for thread in self.threads:
            thread.start()

        self.queue.join()

        for thread in self.threads:
            thread.join()

    def process(self):
        while True:
            if not self.queue.empty():
                item = self.queue.get()
                self.detect_and_save(item)
                self.queue.task_done()
            else:
                break

    def detect_and_save(self, item):
        result_folder = os.path.split(item)[0].replace(".", "./results")
        result_file = os.path.join(result_folder, os.path.split(item)[1])

        try:
            if (
                not os.path.exists(result_file)
                and item.endswith((".jpg", ".jpeg", ".png"))
                and not item.startswith("./results")
            ):
                faces = self.detect_photos(item)
                if len(faces) > 0:
                    self.process_faces(item, result_folder, faces)
                print("Processed file " + item)
        except Exception as e:
            print(
                'File "'
                + os.path.split(item)[1]
                + '" caused exception: '
                + str(e)
            )

    def detect_photos(self, image_name):
        with Image.open(image_name) as image_file:
            image = np.array(image_file)
            detector = MTCNN()
            faces = detector.detect_faces(image)
        return faces

    def process_faces(self, image_path, results_path, faces):
        with Image.open(image_path) as image:
            output_path = os.path.join(
                results_path, os.path.split(image_path)[1]
            )
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            if self.crop is True:
                image = self.crop_largest_face(image, faces)
            else:
                image = self.draw_faces(image, faces)

            image.save(output_path)

    def crop_largest_face(self, image, faces):
        biggest_area = 0
        for face in faces:
            box = face["box"]
            area = box[3] * box[2]
            if area > biggest_area:
                biggest_area = area
                biggest_box = box

        biggest_box[0] = 0 if biggest_box[0] < 0 else biggest_box[0]
        biggest_box[1] = 0 if biggest_box[1] < 0 else biggest_box[1]
        x, y, width, height = biggest_box

        return image.crop((x, y, x + width, y + height))

    def draw_faces(self, image, faces):
        image_draw = ImageDraw.Draw(image)
        for face in faces:
            x, y, width, height = face["box"]
            image_draw.rectangle(
                [x, y, (width + x), (y + height)], fill=None, outline="red"
            )
        return image


def create_dataset_registry(filename):
    if not os.path.exists(filename):
        with open(filename, "w+") as folders_file:
            for root, dirs, files in os.walk("."):
                for name in files:
                    if name.endswith(
                        (".jpg", ".jpeg", ".png")
                    ) and not root.startswith("./results"):
                        folders_file.write(os.path.join(root, name) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images using MTCNN.")
    parser.add_argument(
        "--crop",
        action="store_true",
        help="indicate that largest detected face "
        "will be cropped from image, "
        "instead of marking detected faces.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    filename = "folders.txt"
    create_dataset_registry(filename)

    with open(filename, "r") as folders_file:
        lines = folders_file.readlines()
        lines = [line.rstrip() for line in lines]

        dataset_processor = DatasetProcessor(lines, crop=args.crop)

        dataset_processor.start()


if __name__ == "__main__":
    main()
