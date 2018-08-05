#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    ### your code goes here
    uncleaned_data = [(age, net_worth, net_worth - pred) for age, net_worth, pred in zip(ages, net_worths, predictions)]
    uncleaned_data.sort(key=lambda data_tuple: abs(data_tuple[2]))
    cleaned_data = uncleaned_data[:int(len(uncleaned_data) * 0.9)]

    return cleaned_data

