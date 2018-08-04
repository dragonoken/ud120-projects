#!/usr/bin/python

# For compatibility between python 2 and 3
from __future__ import print_function

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Support Vector Machine Classifiers
# With different C values
clfs = dict((c, SVC(C=c, kernel='rbf')) for c in (10, 100, 1000, 10000))

# Using smaller training dataset to speed up the training process
features_train = features_train[:int(len(features_train) / 100)]
labels_train = labels_train[:int(len(labels_train) / 100)]

accuracy_scores = dict()
for c_value in sorted(clfs):
    # Train the classifier on the training dataset
    # Also, time the duration of the training
    print()
    print("C value :", c_value)
    print()
    print("Training Start...", end='')
    t0 = time()
    clfs[c_value].fit(features_train, labels_train)
    print(" Done!")
    print("Training Finished in ", round(time() - t0, 3), "s", sep='')

    # Get predictions on labels for the test features
    # Also, measure the time it takes to make the predictions
    print()
    print("Making Predictions...", end='')
    t0 = time()
    predictions = clfs[c_value].predict(features_test)
    print(" Done!")
    print("Predictions Made in ", round(time() - t0, 3), "s", sep='')

    # Evaluate the classifier
    print()
    print("With C value", c_value)
    print()
    accuracy_scores[c_value] = accuracy_score(labels_test, predictions)
    print("Accuracy :", accuracy_scores[c_value])
    print()
    print(confusion_matrix(labels_test, predictions))
    print()
    print(classification_report(labels_test, predictions))
    print()

print("\n")
print("Accuracy Score Summary :")
for c_value in sorted(clfs):
    print("C :", c_value, "   Accuracy :", accuracy_scores[c_value])
print()

#########################################################


