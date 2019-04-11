import numpy as np
import pandas as pd

def check_full_duplicates(df):
    if df.shape != df.drop_duplicates().shape:
        print("There are duplicated samples")

def get_meta(train):
    data = []
    for col in train.columns:
        # Defining the role
        if col == 'TARGET':
            role = 'target'
        elif col == 'SK_ID_CURR':
            role = 'id'
        else:
            role = 'input'
        
        # Defining the level
        if col == 'TARGET' or 'FLAG' in col.upper() or 'EMERGENCYSTATE_MODE' in col.upper():
            level = 'binary'
        elif col == 'SK_ID_CURR':
            level = 'nominal'
        elif train[col].dtype == np.float64:
            level = 'interval'
        elif train[col].dtype == np.int64:
            level = 'ordinal'
        elif train[col].dtype == np.object:
            level = 'categorical'
        
        # Initialize keep to True for all variables except for id
        keep = True
        if col == 'SK_ID_CURR':
            keep = False
        
        # Defining the data type 
        dtype = train[col].dtype
        
        # Creating a Dict that contains all the metadata for the variable
        col_dict = {
            'varname': col,
            'role'   : role,
            'level'  : level,
            'keep'   : keep,
            'dtype'  : dtype
        }
        data.append(col_dict)
    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
    meta.set_index('varname', inplace=True)
    return meta

def get_feature_importance_df(feature_importances, 
                              column_names, 
                              top_n=25):
    """Get feature importance data frame.
 
    Parameters
    ----------
    feature_importances : numpy ndarray
        Feature importances computed by an ensemble 
            model like random forest or boosting
    column_names : array-like
        Names of the columns in the same order as feature 
            importances
    top_n : integer
        Number of top features
 
    Returns
    -------
    df : a Pandas data frame
 
    """
     
    imp_dict = dict(zip(column_names, 
                        feature_importances))
    top_features = sorted(imp_dict, 
                          key=imp_dict.get, 
                          reverse=True)[0:top_n]
    top_importances = [imp_dict[feature] for feature 
                          in top_features]
    df = pd.DataFrame(data={'feature': top_features, 
                            'importance': top_importances})
    return df

def get_reduced_important_columns(feat_importances, col_binary, col_categorical):
    col_binary_reduced = []
    col_categorical_reduced = []
    all_columns_reduced = feat_importances.copy()
    for col in col_binary:
        if col in all_columns_reduced:
            col_binary_reduced.append(col)
    
    for col in col_categorical:
        if col in all_columns_reduced:
            col_categorical_reduced.append(col)
    all_columns_reduced = np.insert(all_columns_reduced, 0, values='TARGET')
    
    return col_binary_reduced, col_categorical_reduced, all_columns_reduced

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))