#!/usr/bin/python

# For compatibility between python 2 and 3
from __future__ import print_function

"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from dos2unix import pkl_formatting

data_file_original = "../final_project/final_project_dataset.pkl"
pkl_formatting(data_file_original)
data_file_unix_format = "../final_project/final_project_dataset_unix.pkl"

data_dict = pickle.load(open(data_file_unix_format, "rb"))

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.model_selection import train_test_split
print("Splitting the datasets into train datasets and testing datasets...", end='')
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)
print(" Done!\n")

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
print("Classifier created\n")
print("Training the classifier on the train datasets...", end='')
clf.fit(features_train, labels_train)
print(" Done!\n")
print("Accuracy score on the train dataset :", clf.score(features_train, labels_train))
print("Accuracy score on the test dataset :", clf.score(features_test, labels_test))
print()

from sklearn.metrics import confusion_matrix
pred = clf.predict(features_test)
print("Confusion matrix : (first line : TN FP, second line : FN TP)\n", confusion_matrix(labels_test, pred))
print()

from sklearn.metrics import precision_score, recall_score
print("Precision score :", precision_score(labels_test, pred))
print("Recall score :", recall_score(labels_test, pred))
print()
