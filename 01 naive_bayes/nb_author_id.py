#!/usr/bin/python

# For compatibility between python 2 and 3
from __future__ import print_function

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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

# Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# Train on the training data
# Also, time the duration of the training
print()
print("Training Start...", end='')
t0 = time()
clf.fit(features_train, labels_train)
print(" Done!")
print("Training Finished in ", round(time() - t0, 3), "s", sep='')

# Get predictions on labels for the test features
# Also, measure the time it takes to make the predictions
print()
print("Making Predictions...", end='')
t0 = time()
predictions = clf.predict(features_test)
print(" Done!")
print("Predictions Made in ", round(time() - t0, 3), "s", sep='')

# Evaluate the classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print()
print("Accuracy :", accuracy_score(labels_test, predictions))
print()
print(confusion_matrix(labels_test, predictions))
print()
print(classification_report(labels_test, predictions))
print()

#########################################################
