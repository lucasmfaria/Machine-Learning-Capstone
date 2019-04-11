'''There will be defined preprocessing functions in order to generate input data to the models.
#This makes the notebook (and code) more easily to be understood.'''

import numpy as np
import pandas as pd

#------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------Baseline models preprocessing-----------------------------------------------

#Preprocessing for the Dummy/Naive Classifier
def return_data_dummy_classifier(data):
    X = data.drop('TARGET',axis=1)
    y = data['TARGET']
    return X, y

#Preprocessing for baseline Tree Models which deal with NaNs (XGBoost for example here)
#Label encoding for categorical features
def return_data_baseline_tree(data, col_binary, col_categorical, all_columns):
    df = data[all_columns].copy()
    for col in col_binary:
        if df[col].dtype == np.object:
            df[col] = pd.factorize(df[col], sort=True)[0]
    for col in col_categorical:
        df[col] = pd.factorize(df[col], sort=True)[0]
    X = df.drop('TARGET',axis=1)
    y = df['TARGET']
    return X, y

def return_test_data_baseline_tree(data, col_binary, col_categorical, all_columns):
    df = data[all_columns].copy()
    for col in col_binary:
        if df[col].dtype == np.object:
            df[col] = pd.factorize(df[col], sort=True)[0]
    for col in col_categorical:
        df[col] = pd.factorize(df[col], sort=True)[0]
    return df

#Preprocessing for baseline Tree Models which DO NOT deal with NaNs (RandomForest for example here)
#Label encoding for categorical features
#Fill NaNs with constant
def return_data_baseline_tree_nans(data, col_binary, col_categorical, all_columns):
    df = data[all_columns].copy()
    for col in col_binary:
        if df[col].dtype == np.object:
            df[col] = pd.factorize(df[col], sort=True)[0]
    for col in col_categorical:
        df[col] = pd.factorize(df[col], sort=True)[0]
    df.fillna(-999, inplace=True)
    X = df.drop('TARGET',axis=1)
    y = df['TARGET']
    return X, y

#Preprocessing for linear models which DO NOT deal with NaNs
#One hot encoding for categorical features
def return_data_baseline_linear(data, col_binary, col_categorical, all_columns):
    df = data[all_columns].copy()
    #If the feature is binary, still uses Label encoding in order to reduce the number of columns, as opposed
    #to one hot encoding every column
    for col in col_binary:
        if df[col].dtype == np.object:
            df[col] = pd.factorize(df[col], sort=True)[0]
    df = pd.concat([df.drop(col_categorical, axis=1), pd.get_dummies(df[col_categorical], dummy_na=True)], axis=1)
    df.fillna(-999, inplace=True)
    X = df.drop('TARGET',axis=1)
    y = df['TARGET']
    return X, y

#--------------------------------------------Baseline models preprocessing-----------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------Feature Engineering functions-----------------------------------------------

def create_nan_column(data, new_nan_feature):
    df = data.copy()
    new_nan_column = df[new_nan_feature].isna()
    new_nan_column = new_nan_column.astype(int)
    df[new_nan_feature+"_NAN"] = new_nan_column
    return df

def create_is_k_column(data, new_is_k_feature):
    df = data.copy()
    new_is_k_column = (df[new_is_k_feature].transform(abs) < 0.01)
    new_is_k_column = new_is_k_column.astype(int)
    df[new_is_k_feature+"_IS_K"] = new_is_k_column
    return df

def create_is_anom_column(data, new_is_anom_feature):
    df = data.copy()
    quantiles = df[new_is_anom_feature].quantile([0.25,0.75])
    IQR = quantiles.iloc[1]-quantiles.iloc[0]
    threshold = 1.5*IQR+quantiles.iloc[1]
    new_is_anom_column = (df[new_is_anom_feature].transform(abs) > threshold)
    new_is_anom_column = new_is_anom_column.astype(int)
    df[new_is_anom_feature+"_IS_ANOM"] = new_is_anom_column
    return df

#--------------------------------------------Feature Engineering functions-----------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------Preprocessing for chosen models---------------------------------------------

#Preprocessing for Tree Models which deal with NaNs (XGBoost for example here)
#Label encoding for categorical features
#Adding "feature"_NAN column for input new_nan_features
def return_data_tree(data, col_binary, col_categorical, all_columns, new_nan_features=None, new_is_k_features=None, new_is_anom_features=None):
    df = data[all_columns].copy()
    if new_nan_features is not None:
        for feat in new_nan_features:
            df = create_nan_column(df, feat)
    if new_is_k_features is not None:
        for feat in new_is_k_features:
            df = create_is_k_column(df, feat)
    if new_is_anom_features is not None:
        for feat in new_is_anom_features:
            df = create_is_anom_column(df, feat)
    for col in col_binary:
        if df[col].dtype == np.object:
            df[col] = pd.factorize(df[col], sort=True)[0]
    for col in col_categorical:
        df[col] = pd.factorize(df[col], sort=True)[0]
    X = df.drop('TARGET',axis=1)
    y = df['TARGET']
    return X, y

#Preprocessing for Tree Models which deal with NaNs (XGBoost for example here)
#Label encoding for categorical features
#Adding "feature"_NAN column for input new_nan_features
def return_test_data_tree(data, col_binary, col_categorical, all_columns, new_nan_features=None, new_is_k_features=None, new_is_anom_features=None):
    df = data[all_columns].copy()
    if new_nan_features is not None:
        for feat in new_nan_features:
            df = create_nan_column(df, feat)
    if new_is_k_features is not None:
        for feat in new_is_k_features:
            df = create_is_k_column(df, feat)
    if new_is_anom_features is not None:
        for feat in new_is_anom_features:
            df = create_is_anom_column(df, feat)
    for col in col_binary:
        if df[col].dtype == np.object:
            df[col] = pd.factorize(df[col], sort=True)[0]
    for col in col_categorical:
        df[col] = pd.factorize(df[col], sort=True)[0]
    return df

#--------------------------------------------Preprocessing for chosen models---------------------------------------------
#------------------------------------------------------------------------------------------------------------------------