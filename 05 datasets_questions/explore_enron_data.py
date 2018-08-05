#!/usr/bin/python

# For compatibility between python 2 and 3
from __future__ import print_function

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
"""

import pickle
import sys
sys.path.append("../tools/")
from dos2unix import pkl_formatting
from feature_format import featureFormat, targetFeatureSplit

enron_file = "../final_project/final_project_dataset_unix.pkl"
original_enron_file = "../final_project/final_project_dataset.pkl"
pkl_formatting(original_enron_file)

enron_data = pickle.load(open(enron_file, "rb"))

print("Number of data points :", len(enron_data))
search_keyword = "fastow".upper()
print("Search for", search_keyword, ":", [name for name in enron_data if search_keyword in name])
print()
print("Number of features :", len(next(iter(enron_data.values()))))
print("Features :", list(next(iter(enron_data.values())).keys()))
print()
print("Number of people of interest :", sum(feature['poi'] == 1 for feature in enron_data.values()))
print()
with open("../final_project/poi_names.txt") as f:
    print("Known POIs :", f.readlines()[2:])
print()
print("James Prentice total stock value :", enron_data["PRENTICE JAMES"]["total_stock_value"])
print()
print("Number of emails from Wesley Colwell to POIs :", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])
print()
print("Value of stock options exercised by Jeffrey K Skilling :", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])
print()
print("Money Taken by these 3 POIs :\n")
for poi in ("SKILLING JEFFREY K", "LAY KENNETH L", "FASTOW ANDREW S"):
    print(poi.lower().title(), ":", enron_data[poi]["total_payments"])
print()
print("Number of data points with quantified salaries :", sum(enron_data[data]["salary"] != "NaN" for data in enron_data))
print()
print("Number of data points with known email address :", sum(enron_data[data]["email_address"] != "NaN" for data in enron_data))
print()
print("Number of people missing total payment values :", sum(enron_data[data]["total_payments"] == "NaN" for data in enron_data))
print("Proportion of people missing total payment values :", sum(enron_data[data]["total_payments"] == "NaN" for data in enron_data) / len(enron_data))
print()
print("Number of POIs missing total payment values :", sum(enron_data[data]["total_payments"] == "NaN" and enron_data[data]['poi'] == 1 for data in enron_data))
print("Proportion of POIs missing total payment values :", sum(enron_data[data]["total_payments"] == "NaN" and enron_data[data]['poi'] == 1 for data in enron_data) / sum(enron_data[data]['poi'] == 1 for data in enron_data))
