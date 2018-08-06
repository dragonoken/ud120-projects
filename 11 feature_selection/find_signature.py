#!/usr/bin/python

# For compatibility between python 2 and 3
from __future__ import print_function

import pickle
import numpy
numpy.random.seed(42)
import sys
sys.path.append("../tools/")
from dos2unix import pkl_formatting

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../10 text_learning/your_word_data.pkl" 
authors_file = "../10 text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import model_selection as cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train).toarray()
features_test  = vectorizer.transform(features_test).toarray()
print("Original training features dataset size :", len(features_train))
print("Original training labels dataset size :", len(labels_train))
print()

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]
print("Reduced training features dataset size :", len(features_train))
print("Reduced training labels dataset size :", len(labels_train))
print()

### your code goes here
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
print("Decision Tree Created!\n")
print("Training...", end='')
tree.fit(features_train, labels_train)
print(" Done!\n")
print("Training Accuracy Score :", tree.score(features_train, labels_train))
print("Test Accuracy Score :", tree.score(features_test, labels_test))
print()
top5_important_features = sorted(enumerate(tree.feature_importances_), key=(lambda index_importance_pair: index_importance_pair[1]), reverse=True)[:5]
print("Top 5 importances of features :", [importance for index, importance in top5_important_features])
print("Top 5 important feature numbers :", [index for index, importance in top5_important_features])
print("Features associated with those feature numbers :", [vectorizer.get_feature_names()[index] for index, importance in top5_important_features])
print()
print("End Of Job...")
