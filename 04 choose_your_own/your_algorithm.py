#!/usr/bin/python

# For compatibility between python 2 and 3
from __future__ import print_function

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]


#### initial visualization
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from time import time

# Three Classifiers to choose from!
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
choices = {0:'K Nearest Neighbors Classifier', 1:'Adaptive Boosting Classifier', 2:'Random Forest Classifier'}

chosen_one = 2

def train(classifier, verbose=True):
    if verbose:
        print()
        print("Training start...", end='')
        t0 = time()
    classifier.fit(features_train, labels_train)
    if verbose:
        print(" Done!")
        print("Training finished in ", round(time() - t0, 3), "s", sep='')


def predict(classifier, verbose=True):
    if verbose:
        print()
        print("Making predictions...", end='')
        t0 = time()
    prediction = clf.predict(features_test)
    if verbose:
        print(" Done!")
        print("Predictions made in ", round(time() - t0, 3), "s", sep='')
    return prediction

def evaluate(parameter, prediction, verbose=True):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, prediction)
    if verbose:
        print()
        print("Classifier :", choices[chosen_one])
        print("Parameters : ", end='')
        for param in parameter:
            print(param, "=", parameter[param], end='    ')
        print()
        print("Accuracy :", accuracy)
    return accuracy


if chosen_one == 0:
    print()
    print(choices[chosen_one], "is chosen")
    print()
    weights = ['uniform', 'distance'][1]
    clfs = dict((n, {'classifier':KNeighborsClassifier(n_neighbors=n, weights=weights),
                      'parameter':{'n_neighbors':n, 'weights':weights}}) for n in range(1, 101))
    accuracies = dict()
    for n in clfs:
        clf = clfs[n]['classifier']
        parameter = clfs[n]['parameter']
        print()
        print("Parameters :")
        for param, val in parameter.items():
            print(param, "=", val)

        train(clfs[n]['classifier'])

        predictions = predict(clf)

        accuracies[n] = evaluate(parameter, predictions)

    print("Overall Summary :\n")
    for n in clfs:
        print()
        print("Parameters : ", end='')
        for param, val in clfs[n]['parameter'].items():
            print(param, "=", val, end='    ')
        print("\nAccuracy :", accuracies[n])
    best_n = sorted(accuracies, key=lambda key: accuracies[key])[-1]
    print("\nBest Parameters : ", end='')
    for param, val in clfs[best_n]['parameter'].items():
        print(param, "=", val, end='    ')
    print("Accuracy :", accuracies[best_n])
    clf = clfs[best_n]['classifier']

elif chosen_one == 1:
    print()
    print(choices[chosen_one], "is chosen")
    print()
    learning_rate = 0.8
    clfs = dict((n, {'classifier':AdaBoostClassifier(n_estimators=n, learning_rate=learning_rate, random_state=404),
                    'parameter':{'n_estimators':n, 'learning_rate':learning_rate}}) for n in range(1, 101))
    accuracies = dict()
    for n in clfs:
        clf = clfs[n]['classifier']
        parameter = clfs[n]['parameter']
        print()
        print("Parameters :")
        for param, val in parameter.items():
            print(param, "=", val)

        train(clf)

        predictions = predict(clf)

        accuracies[n] = evaluate(parameter, predictions)

    print("Overall Summary :\n")
    for n in clfs:
        print()
        print("Parameters : ", end='')
        for param, val in clfs[n]['parameter'].items():
            print(param, "=", val, end='    ')
        print("\nAccuracy :", accuracies[n])
    best_n = sorted(accuracies, key=lambda key: accuracies[key])[-1]
    print("\nBest Parameters : ", end='')
    for param, val in clfs[best_n]['parameter'].items():
        print(param, "=", val, end='    ')
    print("Accuracy :", accuracies[best_n])
    clf = clfs[best_n]['classifier']

elif chosen_one == 2:
    print()
    print(choices[chosen_one], "is chosen")
    print()
    n_estimators_vals = list(range(10, 51))
    min_samples_split_vals = list(range(10, 101, 10))
    n_m = []
    for n in n_estimators_vals:
        for m in min_samples_split_vals:
            n_m.append((n, m))
    clfs = dict((i, {'classifier':RandomForestClassifier(n_estimators=n_m_pair[0], min_samples_split=n_m_pair[1], random_state=404),
                     'parameter':{'n_estimators':n_m_pair[0], 'min_samples_split':n_m_pair[1]}}) for i, n_m_pair in enumerate(n_m))
    accuracies = dict()
    for i in clfs:
        clf = clfs[i]['classifier']
        parameter = clfs[i]['parameter']
        print()
        print("Parameters :")
        for param, val in parameter.items():
            print(param, "=", val)

        train(clf, verbose=False)

        predictions = predict(clf, verbose=False)

        accuracies[i] = evaluate(parameter, predictions, verbose=False)

    print("Overall Summary :\n")
    for i in clfs:
        print()
        print("Parameters : ", end='')
        for param, val in clfs[i]['parameter'].items():
            print(param, "=", val, end='    ')
        print("\nAccuracy :", accuracies[i])
    best_i = sorted(accuracies, key=lambda key: accuracies[key])[-1]
    print("\nBest Parameters : ", end='')
    for param, val in clfs[best_i]['parameter'].items():
        print(param, "=", val, end='    ')
    print("Accuracy :", accuracies[best_i])
    clf = clfs[best_i]['classifier']

else:
    raise ValueError("Invalid choice for classifier")


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
