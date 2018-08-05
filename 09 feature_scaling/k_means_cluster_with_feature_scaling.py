#!/usr/bin/python

# For compatibility between python 2 and 3
from __future__ import print_function

"""
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from dos2unix import pkl_formatting



def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### Edit the pkl file into unix format so that it's compatible with python 3 pickle
data_dict_file_original = "../final_project/final_project_dataset.pkl"
pkl_formatting(data_dict_file_original)
data_dict_file_unix_format = "../final_project/final_project_dataset_unix.pkl"
### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load(open(data_dict_file_unix_format, "rb"))
### there's an outlier--remove it!
data_dict.pop("TOTAL", 0)


### the input features we want to use
### can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()

### Rescaling the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(finance_features)
rescaled_finance_features = scaler.transform(finance_features)

test_values = numpy.array([[200000, 1000000]], dtype='float')
print("Salary of $200,000 is rescaled to", scaler.transform(test_values)[0,0])
print("Exercised stock options of $1,000,000 is rescaled to", scaler.transform(test_values)[0,1])
print()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
clusterer = KMeans(n_clusters=2, n_init=10, max_iter=300, random_state=101, n_jobs=1)
clusterer.fit(rescaled_finance_features)
pred = clusterer.predict(rescaled_finance_features)

print("Maximum exercised stock options value :", max([float(data_dict[entry]['exercised_stock_options']) for entry in data_dict if data_dict[entry]['exercised_stock_options'] != "NaN"]))
print("Minimum exercised stock options value :", min([float(data_dict[entry]['exercised_stock_options']) for entry in data_dict if data_dict[entry]['exercised_stock_options'] != "NaN"]))
print()
print("Maximum exercised salary value :", max([float(data_dict[entry]['salary']) for entry in data_dict if data_dict[entry]['salary'] != "NaN"]))
print("Minimum exercised salary value :", min([float(data_dict[entry]['salary']) for entry in data_dict if data_dict[entry]['salary'] != "NaN"]))

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("no predictions object named pred found, no clusters to plot")
