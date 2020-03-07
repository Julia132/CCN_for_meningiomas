import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
#from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split

from imutils import paths
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, \
    classification_report, confusion_matrix, accuracy_score
from imblearn.metrics import sensitivity_specificity_support
from sklearn import metrics

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Convolution2D, ZeroPadding2D
from keras.optimizers import Adam

data = []
labels = []

imagePaths = sorted(list(paths.list_images("part_start")))
ys = sorted(list(paths.list_images("part_finish_0")))
random.seed(100)


for imagePath in imagePaths:
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    image = image / 255.
    image = cv2.resize(image, (84, 84))
    data.append(image)

for y in ys:
    image_y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    image_y = image_y / 255.
    image_y = cv2.resize(image_y, (84, 84))
    labels.append(image_y)

y = [1] * 98 + [0] * 98
trainX, testX, trainY, testY = train_test_split(data, y, test_size=0.25, random_state=100)

model = keras.models.Sequential()

model.add(Conv2D(32, (3, 3),  padding='same', input_shape=(84, 84, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

#model.add(concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

#up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

#up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

#up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(1, (1, 1), activation='sigmoid'))

#model = Model(inputs=[inputs], outputs=[conv10]) подумать, как собирается модель
model.summary()

model.compile(optimizer=Adam(lr=1e-4), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
EPOCHS = 120

trainX = np.expand_dims(np.array(trainX), axis=3)
testX = np.expand_dims(np.array(testX), axis=3)


H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32, verbose=2)
print("[INFO] evaluating network...")

predictions = model.predict_classes(testX, batch_size=32)
print(testY)
print(predictions)
matrix_metrics = classification_report(testY, predictions)
print(matrix_metrics)

precision, recall, fscore, support = precision_recall_fscore_support(testY, predictions)
_, specificity, _ = sensitivity_specificity_support(testY, predictions)
print('Accuracy', accuracy_score(testY, predictions))
print('binary precision value', precision[0])
print('binary recall value', recall[0])
print('binary fscore value', fscore[0])
print('binary specificity value', specificity[0])
#print(confusion_matrix(testY.round(), predictions))
#print(classification_report(testY, predictions))
dashList = [(5, 2), (4, 10), (3, 3, 2, 2), (5, 2, 20, 2)]

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


plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(testY, predictions)
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

plt.figure(figsize=(8, 8))
recall = metrics.recall_score(testY, predictions, average=None)
specificity = recall[0]
precision, recall, thresholds = metrics.precision_recall_curve(testY, predictions)
plt.plot(recall, precision)
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Curve dependent Precision и Recall of threshold")
plt.legend(loc='best')
plt.show()