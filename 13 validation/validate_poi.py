#!/usr/bin/python

# For compatibility between python 2 and 3
from __future__ import print_function

"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from dos2unix import pkl_formatting

data_file_original = "../final_project/final_project_dataset.pkl"
pkl_formatting(data_file_original)
data_file_unix_format = "../final_project/final_project_dataset_unix.pkl"

data_dict = pickle.load(open(data_file_unix_format, "rb") )
print("Dataset loaded\n")

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

print("Formatting and splitting the dataset...", end='')
data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)
print(" Done!\n")

### it's all yours from here forward!
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
print("Classifier created\n")
print("Training on the entire features and labels datasets...", end='')
clf.fit(features, labels)
print(" Done!\n")
print("Accuracy score :", clf.score(features, labels))
print()

from sklearn.model_selection import train_test_split
print("Splitting the datasets into train datasets and testing datasets...", end='')
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)
print(" Done!\n")
print("Re-training the classifier on the train datasets...", end='')
clf.fit(features_train, labels_train)
print(" Done!\n")
print("Accuracy score on the train dataset :", clf.score(features_train, labels_train))
print("Accuracy score on the test dataset :", clf.score(features_test, labels_test))
