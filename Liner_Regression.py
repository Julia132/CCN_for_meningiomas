import pre_processing
import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, \
    classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.metrics import sensitivity_specificity_support
from sklearn import metrics


data, y = pre_processing.get_image("scan/D_scan", "scan/D_scan_output")
data = np.array(data)
y = np.array(y)

trainX, testX, trainY, testY = train_test_split(data, y, test_size=0.25, random_state=100)


def rea_shep(sample):

    n_samples, nx, ny = sample.shape
    sample = sample.reshape((n_samples, nx*ny))
    sample = sample.flatten()
    sample = sample.reshape(-1, 1)

    return sample


trainX = rea_shep(trainX)
trainY = rea_shep(trainY)
testX = rea_shep(testX)
testY = rea_shep(testY)

print(trainX.shape)
print(trainY.shape)

classifier = LinearRegression()
classifier.fit(trainX, trainY)

y_pred = classifier.predict(testX)

testY = np.where(testY > 0.5, 1, 0)


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


optimal_threshold = get_optimal_threshold(y_pred, testY)

print('optimal_threshold', optimal_threshold)

predictions = np.where(y_pred > optimal_threshold, 1, 0)

# r_sq = classifier.score(testY, y_pred)
# print('coefficient of determination:', r_sq)
# print('intercept:', classifier.intercept_)
# print('slope:', classifier.coef_)
# print('predicted response:', y_pred, sep='\n')

precision, recall, f_score, support = precision_recall_fscore_support(testY, predictions)
_, specificity, _ = sensitivity_specificity_support(testY, predictions)
print('Accuracy', accuracy_score(testY, predictions))
print('binary precision value', precision[1])
print('binary recall value', recall[1])
print('binary f_score value', f_score[1])
print('binary specificity value', specificity[1])

print(classification_report(testY, predictions))
print(confusion_matrix(testY, predictions))


plt.figure(figsize=(8, 6))
fpr, tpr, threshold = roc_curve(testY, predictions)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('CNN', roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Correctly defined pixels')
plt.ylabel('Fallaciously defined pixels')
plt.legend(loc=0, fontsize='small')
plt.title("ROC - curve")
plt.show()

plt.figure(figsize=(8, 8))
recall = metrics.recall_score(testY, predictions, average=None)

precision, _, thresholds = metrics.precision_recall_curve(testY, predictions)
plt.plot(recall, precision)
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Curve dependent Precision и Recall of threshold")
plt.legend(loc='best')
plt.show()