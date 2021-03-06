import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
seed_value = 0
np.random.seed(seed_value)
import pandas as pd
import random

import pre_processing
from os import listdir
from os.path import isfile, join

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
#from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split

from imutils import paths
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, \
    classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.metrics import sensitivity_specificity_support
from sklearn import metrics

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Convolution2D, ZeroPadding2D
from keras.optimizers import Adam, Adadelta, SGD


def get_optimal_threshold(prediction, testY):

    j = -1e-7
    best_item = []

    for item in np.linspace(np.min(prediction), np.max(prediction), 1000):

        pred = np.where(prediction > item, 1, 0)
        metric = f1_score(testY, pred)

        if metric > j:

            j = metric
            best_item.append(item)

    return best_item[-1]


def model():

    # data, hand = pre_processing.get_image("result_segment_input", "result_segment_output")
    # data, hand = pre_processing.get_image("test_input", "test_output")
    data, hand = pre_processing.get_image("scan/T2_scan", "scan/T2_scan_output")


    trainX, testX, trainY, testY = train_test_split(data, hand, test_size=0.25, random_state=100)

    model = keras.models.Sequential()

    model.add(Conv2D(32, (3, 3),  padding='same', input_shape=(64, 64, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (6, 6), activation='relu', strides=(1, 1), padding='same'))

    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (6, 6), activation='relu', strides=(1, 1), padding='same'))
    # model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (9, 9), activation='relu', strides=(1, 1), padding='same'))
    # model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))

    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (9, 9), activation='relu', strides=(1, 1), padding='same'))
    # model.add(Conv2D(256, (6, 6), activation='relu', strides=(1, 1), padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(384, (6, 6), activation='relu', strides=(1, 1), padding='same'))
    # model.add(Conv2D(384, (6, 6), activation='relu', strides=(1, 1), padding='same'))

    #model.add(concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1))
    # model.add(Conv2D(256, (6, 6), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(256, (9, 9), activation='relu', strides=(1, 1), padding='same'))

    #up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    model.add(Conv2D(128, (9, 9), activation='relu', strides=(1, 1), padding='same'))
    # model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))

    #up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    # model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(64, (6, 6), activation='relu', strides=(1, 1), padding='same'))

    #up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))

    model.add(Conv2D(1, (1, 1), strides=(1, 1), activation='relu'))

    #model = Model(inputs=[inputs], outputs=[conv10]) подумать, как собирается модель
    model.summary()

    model.compile(optimizer=Adam(lr=1e-5), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    EPOCHS = 5

    trainX = np.expand_dims(np.array(trainX), axis=3)
    testX = np.expand_dims(np.array(testX), axis=3)

    trainY = np.expand_dims(np.array(trainY), axis=3)
    testY = np.expand_dims(np.array(testY), axis=3)

    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)

    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32, verbose=2)
    print("[INFO] evaluating network...")

    predictions = model.predict(testX, batch_size=32)

    # from sklearn import preprocessing
    # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # for i in range(predictions.shape[0]):
    #      cv2.imshow("frame", testY[i, :, :, 0])
    #      cv2.waitKey(0)
    #      cv2.imshow("frame", scaler.fit_transform(predictions[i, :, :, 0]))
    #      cv2.waitKey(0)

    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(testY.flatten(), predictions.flatten())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('CNN', roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Correctly defined neoplasms')
    plt.ylabel('Fallaciously defined neoplasms')
    plt.legend(loc=0, fontsize='small')
    plt.title("ROC - curve")
    plt.show()

    #threshold = (np.min(predictions) + np.max(predictions)) / 2.0
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx]
    testY = np.where(testY > 0.5, 1, 0)

    optimal_threshold = get_optimal_threshold(predictions.flatten(), testY.flatten())

    print('optimal_threshold', optimal_threshold)

    predictions = np.where(predictions > optimal_threshold, 1, 0)

    precision, recall, fscore, support = precision_recall_fscore_support(testY.flatten(), predictions.flatten())
    _, specificity, _ = sensitivity_specificity_support(testY.flatten(), predictions.flatten())
    print('Accuracy', accuracy_score(testY.flatten(), predictions.flatten()))
    print('binary precision value', precision[1])
    print('binary recall value', recall[1])
    print('binary fscore value', fscore[1])
    print('binary specificity value', specificity[1])

    print(classification_report(testY.flatten(), predictions.flatten()))

    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss", linestyle='--')
    plt.plot(N, H.history["val_loss"], label="val_loss", linestyle='-.')
    plt.plot(N, H.history["acc"], label="train_acc", linestyle= ':')
    plt.plot(N, H.history["val_acc"], label="val_acc",  linestyle='-')
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig("Training Loss and Accuracy.pdf")
    plt.show()


    plt.figure(figsize=(8, 8))
    recall = metrics.recall_score(testY.flatten(), predictions.flatten(), average=None)

    precision, recall, thresholds = metrics.precision_recall_curve(testY.flatten(), predictions.flatten())
    plt.plot(recall, precision)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Curve dependent Precision и Recall of threshold")
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    model()
