import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import r2_score


MODEL_TYPES = {"linear": LinearRegression, "svm": SVC, "krr": KernelRidge, "ada": AdaBoostRegressor}

def rounded_r2_score(y_true, y_pred):
    """
    Compute the R^2 score after rounding the predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True values for the target variable.
    y_pred : np.ndarray
        Predicted values before rounding.

    Returns
    -------
    r2 : float
        R^2 (coefficient of determination) regression score function.
    """
    rounded_y_pred = np.floor(y_pred + 0.5)
    return r2_score(y_true, rounded_y_pred)