#!/usr/bin/python

# For compatibility between python 2 and 3
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import sys
import pickle
import numpy as np
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from dos2unix import pkl_formatting

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
    ### Label
    'poi'

    ### Financial Features
    , 'salary'
    , 'deferral_payments'
    , 'total_payments'
    , 'loan_advances'
    , 'bonus'
    , 'restricted_stock_deferred'
    , 'deferred_income'
    , 'total_stock_value'
    , 'expenses'
    , 'exercised_stock_options'
    , 'other'
    , 'long_term_incentive'
    , 'restricted_stock'
    , 'director_fees'

    ### Email Features
    , 'to_messages'
    # , 'email_address'
    , 'from_poi_to_this_person'
    , 'from_messages'
    , 'from_this_person_to_poi'
    , 'shared_receipt_with_poi'

    ### Custom Features (Features I created)
    , 'from_poi_ratio'
    , 'to_poi_ratio'
]

### Load the dictionary containing the dataset
dataset_file_original = "final_project_dataset.pkl"
pkl_formatting(dataset_file_original)
dataset_file_unix_format = "final_project_dataset_unix.pkl"
with open(dataset_file_unix_format, "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)

### Task 3: Create new feature(s)
for person in data_dict:
    if data_dict[person]['from_poi_to_this_person'] == 'NaN' or data_dict[person]['from_messages'] == 'NaN':
        data_dict[person]['from_poi_ratio'] = 0
    else:
        data_dict[person]['from_poi_ratio'] = float(data_dict[person]['from_poi_to_this_person']) / float(data_dict[person]['from_messages'])

    if data_dict[person]['from_this_person_to_poi'] == 'NaN' or data_dict[person]['to_messages'] == 'NaN':
        data_dict[person]['to_poi_ratio'] = 0
    else:
        data_dict[person]['to_poi_ratio'] = float(data_dict[person]['from_this_person_to_poi']) / float(data_dict[person]['to_messages'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

features = np.array(features)
labels = np.array(labels)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

print("Reducing the dimensionality of the features using PCA...", end='')
t0 = time()
pca = PCA(n_components='mle', whiten=True, random_state=42)
features = pca.fit_transform(features)
print(" Done!")
print("PCA finished in %0.3fs\n" % (time() - t0))
print("Principal components explained variance ratios :\n", pca.explained_variance_ratio_)
print()

print("Creating classifiers...", end='')
t0 = time()
param_grid_svc = {'C':[c for c in range(1, 21, 1)], 'gamma':['auto'] + [g / 1000 for g in range(1, 21, 1)]}
param_grid_logreg = {'C':[c for c in range(10, 2001, 10)]}
param_grid_randforest = {'n_estimators':[n for n in range(1, 21, 1)], 'min_samples_split':[s for s in range(2, 21, 1)]}

clf_nb = GridSearchCV(GaussianNB(), {}, scoring='f1', verbose=3)
clf_svc = GridSearchCV(SVC(random_state=42), param_grid_svc, scoring='f1', verbose=3)
clf_logreg = GridSearchCV(LogisticRegression(random_state=42), param_grid_logreg, scoring='f1', verbose=3)
clf_randforest = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_randforest, scoring='f1', verbose=3)
print(" Done!")
print("Classifiers created in %0.3fs\n" % (time() - t0))

clfs = {
    'Naive Bayes':clf_nb
   ,'Support Vector Machine':clf_svc
   ,'Logistic Regression':clf_logreg
   ,'Random Forest':clf_randforest
}

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import train_test_split
print("Splitting the dataset into train dataset and test dataset...", end='')
t0 = time()
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
print(" Done!")
print("Data split completed in %0.3fs\n" % (time() - t0))

print("Training all classifiers\n")
t0 = time()
for classifier in clfs:
    clfs[classifier].fit(features_train, labels_train)
print("\nTraining finished in %0.3fs\n" % (time() - t0))

clf_scores = dict()
for classifier in clfs:
    print("-----------------------------------------------------")
    print("Classifier :", classifier)
    print()
    print("Best parameters :", clfs[classifier].best_params_)
    print()

    pred_train = clfs[classifier].predict(features_train)
    pred_test = clfs[classifier].predict(features_test)

    print("<Train Summary>\n")
    print("f1 score :", f1_score(labels_train, pred_train))
    print("Precision score :", precision_score(labels_train, pred_train))
    print("Recall score :", recall_score(labels_train, pred_train))
    print("Confusion matrix :\n", confusion_matrix(labels_train, pred_train))
    print("Classification report :\n", classification_report(labels_train, pred_train))

    print("<Test Summary>\n")
    print("f1 score :", f1_score(labels_test, pred_test))
    print("Precision score :", precision_score(labels_test, pred_test))
    print("Recall score :", recall_score(labels_test, pred_test))
    print("Confusion matrix :\n", confusion_matrix(labels_test, pred_test))
    print("Classification report :\n", classification_report(labels_test, pred_test))

    clf_scores[classifier] = dict()
    clf_scores[classifier]['f1'] = f1_score(labels_test, pred_test)
    clf_scores[classifier]['precision'] = precision_score(labels_test, pred_test)
    clf_scores[classifier]['recall'] = recall_score(labels_test, pred_test)
    print("-----------------------------------------------------\n")


# Picking up a classifier with the best f1 score for the test
best_classifier = max(clf_scores, key=(lambda classifier: clf_scores[classifier]['f1']))
clf = clfs[best_classifier].best_estimator_
print("Best estimator :", clf)
print("f1 score :", clf_scores[best_classifier]['f1'])
print("precision score :", clf_scores[best_classifier]['precision'])
print("recall score :", clf_scores[best_classifier]['recall'])

# Manual Selection
choices = {'1':clfs['Naive Bayes'], '2':clfs['Support Vector Machine'], '3':clfs['Logistic Regression'], '4':clfs['Random Forest']}
choice = input("\n1 : Naive Bayes\n2 : SVM\n3 : Logistic Regression\n4 : Random Forest\n\nChoose one : ").strip()
while choice not in ('1', '2', '3', '4'):
    choice = input("\n1 : Naive Bayes\n2 : SVM\n3 : Logistic Regression\n4 : Random Forest\n\nChoose one : ").strip()
clf = choices[choice].best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
