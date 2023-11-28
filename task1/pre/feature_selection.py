import numpy as np
from sklearn.linear_model import Lasso

def feature_selection_pearson_correlation(input_mat, threshold=0.8, copy=False, **kwargs):
    """
    Perform feature selection based on Pearson's correlation analysis

    Parameters
    ----------
    input_mat : np.array
        Input matrix to perform feature selection
    threshold: float, default=0.8
        Threshold for pearson's correlation to remove highly correlated features
    copy: bool, default=False
        Whether to filter features on original matrix or create a copy

    Return
    ------
    np.array
        Matrix with filtered features

    """
    drop = True
    mat = input_mat.copy() if copy else input_mat
    col_indices = np.array(range(mat.shape[1]))
    absolute_cols_to_keep = []
    while drop:
        correlation_mat = np.corrcoef(mat, rowvar=False)
        np.fill_diagonal(correlation_mat, 0)
        indices_to_drop = np.where(correlation_mat > threshold)
        if len(indices_to_drop[0]) == 0:
            drop = False
        else:
            cols_to_drop = []
            for idx1, idx2 in zip(*indices_to_drop):
                if idx1 not in cols_to_drop and idx2 not in cols_to_drop:
                    cols_to_drop.append(idx1)
            cols_to_keep = set(list(range(correlation_mat.shape[1]))) - set(cols_to_drop)
            cols_to_keep = list(cols_to_keep)
            absolute_cols_to_keep += col_indices[cols_to_keep]
            cols_indices = cols_indices[cols_to_keep]
            mat = mat[:, list(cols_to_keep)]
    print(f"Number of selected features: {len(absolute_cols_to_keep)}")
    return mat

def feature_selection_lasso(input_mat, label, alpha=0.1, **kwargs):
    """
    Perform feature selection based on Lasso regression

    Parameters
    ----------
    input_mat : np.array
        Input matrix to perform feature selection
    label: np.array
        Label corresponding to `input_mat` for regression
    alpha: float, default=0.1
        Threshold for Lasso regularization 

    Return
    ------
    tuple[np.array, np.array]
        Matrix with filtered features and indices of selected features

    """
    lasso = Lasso(alpha=alpha)
    lasso.fit(input_mat, label)

    selected_features = np.where(lasso.coef_ != 0)[0]

    mat = input_mat[:, selected_features]
    print(f"Number of selected features: {len(selected_features)}")
    return mat, selected_features
