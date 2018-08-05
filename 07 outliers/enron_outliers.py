#!/usr/bin/python

# For compatibility between python 2 and 3
from __future__ import print_function

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from dos2unix import pkl_formatting


### read in data dictionary, convert to numpy array
data_dict_file_original = "../final_project/final_project_dataset.pkl"
pkl_formatting(data_dict_file_original)
data_dict_file_unix_format = "../final_project/final_project_dataset_unix.pkl"

data_dict = pickle.load(open(data_dict_file_unix_format, "rb"))
data_dict.pop("TOTAL", 0)
target_feature = "salary"
input_feature = "bonus"
features = [target_feature, input_feature]
data = featureFormat(data_dict, features)


### your code below
plt = matplotlib.pyplot
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)
plt.xlabel("salary")
plt.ylabel("bonus")

print("Outliers with at least $5M of bonus and $1M of salary :",
      [outlier for outlier in data_dict if float(data_dict[outlier]["bonus"]) > 5e+6 and float(data_dict[outlier]["salary"]) > 1e+6])
plt.show()
