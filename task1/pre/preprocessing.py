from task1.pre.feature_selection import (
    feature_selection_lasso,
    feature_selection_pearson_correlation,
)
from task1.pre.imputation import (
    impute_dataframe,
    knn_impute_dataframe,
    iterative_impute_dataframe,
    impute_data_with_mice_and_simple,
)
from task1.pre.outliers_detection import outliers_z_score, outliers_dbscan, outliers_isolation_forest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class DataPreprocessor:
    OUTLIERS_DETECTION = {"z_score": outliers_z_score, "isolation_forest": outliers_isolation_forest, "dbscan": outliers_dbscan}
    IMPUTATION = {
        "knn": knn_impute_dataframe,
        "simple": impute_dataframe,
        "iterative": iterative_impute_dataframe,
        "mice": impute_data_with_mice_and_simple
    }
    FEATURE_SELECTION = {
        "lasso": feature_selection_lasso,
        "pearson": feature_selection_pearson_correlation,
    }

    def __init__(self, preprocessing_config):
        self.preprocessing_config = preprocessing_config

    def preprocess(self, X_train_df, y_train, X_valid_df):
        X_df = pd.concat([X_train_df, X_valid_df])
        train_length = len(X_train_df)
        if self.preprocessing_config['scale']:
            scaler = StandardScaler()
            X_mat_scaled = scaler.fit_transform(X_df)
            X_df = pd.DataFrame(X_mat_scaled, columns=X_df.columns)

        # Always do imputation first
        imputation_method = self.preprocessing_config['imputation']['method']
        imputation_config = self.preprocessing_config['imputation']['arguments']
        X_df = self.IMPUTATION[imputation_method](X_df, **imputation_config)
        X_mat = X_df.to_numpy()

        for step in self.preprocessing_config['execution_order']:
            if step == 'imputation':
                continue
            method = self.preprocessing_config[step]['method']
            method_config = self.preprocessing_config[step]['arguments']
            if step == 'feature_selection':
                X_train_mat = X_mat[:train_length]
                X_train_mat, selected_features = getattr(self, step.upper())[method](X_train_mat, label=y_train, **method_config)
                X_valid_mat = X_mat[train_length:]
                X_valid_mat = X_valid_mat[:, selected_features]
                X_mat = np.concatenate((X_train_mat, X_valid_mat), axis=0)
            elif step == 'outliers_detection':
                y_length = len(y_train)
                X_valid_mat = X_mat[train_length:]
                X_train_mat, y_train = getattr(self, step.upper())[method](X_mat[:train_length], X_valid_mat, y_train, **method_config)
                train_length = train_length - y_length + len(y_train)
                X_mat = np.concatenate((X_train_mat, X_valid_mat), axis=0)
            else:
                X_mat = getattr(self, step.upper())[method](X_mat, **method_config)

        return X_mat[:train_length], X_mat[train_length:], y_train 