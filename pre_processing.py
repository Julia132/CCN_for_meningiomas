import cv2
import numpy as np
import csv
import math
import os

from os import listdir
from os.path import isfile, join
from imutils import paths


path_source = sorted(list(paths.list_images("source_datum")))

path_handling = sorted(list(paths.list_images("manual_handling")))

path_result = sorted(list(paths.list_images("result_segment")))


data = []

hand = []


def rename(path_source, path_handling):

    only_source = [f for f in (path_source) if (join(path_source, f))]
    only_handling = [f for f in listdir(path_handling) if isfile(join(path_handling, f))]

    for filename in only_source:

        os.rename(os.path.join(path_source, filename), os.path.join(path_source, filename.replace(' ', '_')))
        os.rename(os.path.join(path_source, filename), os.path.join(path_source, filename.replace(',', '.')))

    for filename in only_handling:

        os.rename(os.path.join(path_handling, filename), os.path.join(path_handling, filename.replace(' ', '_')))
        os.rename(os.path.join(path_handling, filename), os.path.join(path_handling, filename.replace(',', '.')))

    return path_handling, path_source

rename(path_handling, path_source)

#def predata(path_source, path_handling, data, hand):
image = np.empty(len(path_handling), dtype=object)

for item in range(0, len(only_source)):

    image[item] = cv2.imread(join(path_handling, only_source[item]))
    gray = cv2.cvtColor(image[item], cv2.COLOR_BGR2GRAY)

    cv2.imshow('dfghjk', gray)
    image = image / 255.
    image = cv2.resize(image, (84, 84))
    data.append(image)
    print(path_source)

for path in (path_handling):

    neoplasm = cv2.imread(path, cv2.IMREAD_COLOR)
    neoplasm = neoplasm / 255.
    neoplasm = cv2.resize(neoplasm, (84, 84))
    height, width = neoplasm.shape

    for x in range(0, width):

        for y in range(0, height):

            if neoplasm[x, y, 0] != 0 and neoplasm[x, y, 1] != 0 and neoplasm[x, y, 2] != 0:

                neoplasm[x, y, 0] = 255
                neoplasm[x, y, 1] = 255
                neoplasm[x, y, 2] = 255

    hand.append(neoplasm)

    #return hand, data

