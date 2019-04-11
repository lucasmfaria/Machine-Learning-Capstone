import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from functions.utils import sigmoid
from sklearn.metrics import roc_auc_score
import xgboost
import matplotlib.pyplot as plt

def cross_val_model(X, y, model, n_splits=3, use_random_state=True):
    X = np.array(X)
    y = np.array(y)
    if use_random_state:
        sfkf = StratifiedKFold(n_splits=n_splits, random_state=10)
    else:
        sfkf = StratifiedKFold(n_splits=n_splits)
    cross_score = cross_val_score(model, X, y, cv=sfkf, scoring='roc_auc')
    return cross_score

def modelfit(xgb_model, X_train, y_train, X_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    X_train_df = X_train.copy()
    y_train_df = y_train.copy()
    
    if useTrainCV:
        xgb_params = xgb_model.get_xgb_params()
        train_dmatrix = xgboost.DMatrix(X_train_df.values, label=y_train_df.values, feature_names=X_train_df.columns)
        xgbcv_results = xgboost.cv(xgb_params, train_dmatrix, num_boost_round=xgb_model.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', stratified=True, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        #Setting the optimal n_estimators values found by early_stopping with cross validation:
        xgb_model.set_params(n_estimators=xgbcv_results.shape[0])
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    #Fit the algorithm on the data
    xgb_model.fit(X_train_df, y_train_df, eval_metric='auc', eval_set=eval_set, verbose=False)
        
    #Predict training set:
    y_train_predictions = xgb_model.predict(X_train_df)
    y_train_predprob = xgb_model.predict_proba(X_train_df)[:,1]
    
    y_test_predictions = xgb_model.predict(X_test)
    y_test_predprob = xgb_model.predict_proba(X_test)[:,1]
        
    #Print model report:
    print("\nModel Report")
    if useTrainCV:
        print("Number of estimators (early stopping and cross-validation): ", xgbcv_results.shape[0])
    print("AUC Score (Train): %f" % roc_auc_score(y_train_df, y_train_predprob))
    print("AUC Score (Test): %f" % roc_auc_score(y_test, y_test_predprob))
                    
    feat_imp = pd.Series(xgb_model.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

def nested_cross_val_model(X, y, grid_model, n_splits_outter=3, use_random_state=True):
    X = np.array(X)
    y = np.array(y)
    if use_random_state:
        folds = list(StratifiedKFold(n_splits=n_splits_outter, random_state=10).split(X, y))
    else:
        folds = list(StratifiedKFold(n_splits=n_splits_outter).split(X, y))
    cross_scores = []
    best_estimators = []
    best_params = []
    
    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_holdout = X[test_idx]
        y_holdout = y[test_idx]
        
        print ("Fit %s - outter fold %d" % (str(grid_model).split('(')[0], j+1))
        
        grid_model.fit(X_train, y_train)
        
        print("------------------------------Best Estimator:-----------------------------")
        print(grid_model.best_estimator_)
        print("-----------------------------Best Inner Score:----------------------------")
        print("Mean cross-validated score of the best_estimator:")
        print(grid_model.best_score_)
        print("-----------------------------Best Parameters:-----------------------------")
        print(grid_model.best_params_)
        print("--------------------------Outter Score Estimation:------------------------")
        prob_preds = grid_model.predict_proba(X_holdout)
        y_holdout_score = prob_preds[:, 1]
        y_holdout_probs = sigmoid(y_holdout_score)
        outter_score = roc_auc_score(y_holdout, y_holdout_probs)
        print(outter_score)
        print("")
        cross_scores.append(outter_score)
        best_estimators.append(grid_model.best_estimator_)
        best_params.append(grid_model.best_params_)
    
    return cross_scores, best_estimators, best_params

def plot_xgb_learning_curve(xgb_model):
    results = xgb_model.evals_result()
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)
    # plot AUC
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax.legend()
    plt.ylabel('AUC')
    plt.title('XGBoost AUC')
    plt.show()