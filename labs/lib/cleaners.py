# -*- coding: utf-8 -*-
import numpy as np

def normalize_features(x):
    weights = 1

def prepare_data(x, replace_with="median", standardize=True, outliers=True, low=1, high=99, log = False):
    # filling up missing values
    for i in range(0, len(x[1, :])):
        if replace_with == "mean":
            # Calculating mean of each column (without considering missing values)
            replace_val = np.mean(x[x[:, i] != -999, i])
            if log and i == 0:
                print("Missing values are replaced by the average value")
        elif replace_with == "median":
            # Calculating median of each column (without considering missing values)
            replace_val = np.median(x[x[:, i] != -999, i])
            if log and i == 0:
                print("Missing values are replaced by the median value")
        # Replacing missing values
        x[:, i] = np.where(x[:, i] == -999, replace_val, x[:, i])

    if (outliers):
        low_percentile = np.percentile(x, low, axis=0)
        high_percentile = np.percentile(x, high, axis=0)
        for i in range(0, len(x[1, :])):
            if replace_with == "mean":
                # Calculating average of each column
                replace_val = np.mean(x[:, i])
                if log and i == 0:
                    print("Outliers values (values under",low, "percentile and over", high, "percentile) are replaced by the average value")
            elif replace_with == "median":
                # Calculating median of each column
                replace_val = np.median(x[:, i])
                if log and i == 0:
                    print("Outliers values (values under",low, "percentile and over", high, "percentile) are replaced by the median value")
            # Replacing outliers
            x[:, i] = np.where(x[:, i] < low_percentile[i], replace_val, x[:, i])
            x[:, i] = np.where(x[:, i] > high_percentile[i], replace_val, x[:, i])
    if (standardize):
        if log:
            print("Data are normalized")
        x_ = np.copy(x)
        x = (np.array([standardize_column(column) for column in x_.T])).T
    return x


def standardize_column(col):
    mean = np.mean(col)
    std = np.std(col)
    if np.isnan(std) or std == 0:
        return np.zeros(col.shape)
    col = (col-mean)/std
    return col

def standardize_column_mean(col):
    filtered_col = col[col != -999]
    mean = np.mean(filtered_col)
    std = np.std(filtered_col)
    if np.isnan(std) or std == 0:
        return np.zeros(col.shape)
    col[col == -999] = mean
    col = (col-mean)/std
    #_test_ = (filtered_col - mean)/std
    #print(f"test: mean={np.mean(_test_)} std={np.std(_test_)}")
    return col

def standardize_mean(x_):
    x = np.copy(x_)
    tx = np.array([standardize_column_mean(column) for column in x.T])
    return tx.T
