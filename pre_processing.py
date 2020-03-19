import cv2
import numpy as np
import os

from os import listdir
from os.path import isfile, join

path_source = "source_datum"

path_handling = "manual_handling"

path_result_input = "result_segment_input"

path_result_output = 'result_segment_output'


def rename(*args):

    for dir_name in args:

        for filename in [f for f in listdir(dir_name) if isfile(join(dir_name, f))]:

            os.rename(
                os.path.join(dir_name, filename), os.path.join(dir_name, filename.replace(' ', '_').replace(',', '.'))
            )


def predata(path_input, path_output):

    for filename in [f for f in listdir(path_input) if isfile(join(path_input, f))]:

        image = cv2.imread(join(path_input, filename))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (64, 64))

        cv2.imwrite(join(path_output, filename), image)


def change_color(dir_name):

    for filename in [f for f in listdir(dir_name) if isfile(join(dir_name, f))]:

        neoplasm = cv2.imread(join(dir_name, filename))
        R = np.mean(neoplasm)

        if R >= 127:

            neoplasm = 255 - neoplasm

        thresh, neoplasm = cv2.threshold(neoplasm.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)

        cv2.imwrite(join(dir_name, filename), neoplasm)


def get_image(dir_input, dir_output):

    data = []

    hand = []

    for filename in [f for f in listdir(dir_input) if isfile(join(dir_input, f))]:

        image = cv2.imread(join(dir_input, filename))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image / 255.

        data.append(image)

    for filename in [f for f in listdir(dir_output) if isfile(join(dir_output, f))]:

        image = cv2.imread(join(dir_output, filename))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image / 255.

        hand.append(image)

    return data, hand


def main():
    rename(path_handling, path_source)
    predata(path_handling, path_result_output)
    predata(path_source, path_result_input)
    change_color(path_result_output)


if __name__ == "__main__":
    main()

