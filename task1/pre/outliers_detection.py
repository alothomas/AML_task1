import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

def outliers_dbscan(train_X, valid_X ,label, **kwargs):
    """
    Detect and process outliers based on DBSCAN

    Parameters
    ----------
    input_mat : np.array
        Input matrix to perform outliers detection

    Return
    ------
    np.array
        Matrix with processed outliers

    """
    input_mat = np.concatenate((train_X, valid_X), axis=0)
    dbscan = DBSCAN(**kwargs)
    dbscan.fit(input_mat)
    y_pred = dbscan.labels_[:len(train_X)]
    train_X = train_X[y_pred != -1]
    label = label[y_pred != -1]
    return train_X, label

def outliers_z_score(input_mat, **kwargs):
    """
    Detect and process outliers based on Z-score

    Parameters
    ----------
    input_mat : np.array
        Input matrix to perform outliers detection

    Return
    ------
    np.array
        Matrix with processed outliers

    """
    mean = input_mat.mean(axis=0)
    std = input_mat.std(axis=0)
    std = input_mat.std(axis=0)
    lower_limit = mean - 3*std
    upper_limit = mean + 3*std

    input_mat[input_mat < lower_limit] = np.tile(lower_limit, (input_mat.shape[0], 1))[input_mat < lower_limit]
    input_mat[input_mat > upper_limit] = np.tile(upper_limit, (input_mat.shape[0], 1))[input_mat > upper_limit]    
    return input_mat, None


def outliers_isolation_forest(train_X, valid_X, label, **kwargs):
    """
    Detect and process outliers based on Isolation Forest

    Parameters
    ----------
    input_mat : np.array
        Input matrix to perform outliers detection

    Return
    ------
    np.array
        Matrix with processed outliers

    """
    input_mat = np.concatenate((train_X, valid_X), axis=0)
    clf = IsolationForest(**kwargs)
    clf.fit(input_mat)
    y_pred = clf.predict(train_X)
    train_X = train_X[y_pred == 1]
    label = label[y_pred == 1]
    return train_X, label