import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor


def impute_dataframe(df, strategy='most_frequent', **kwargs):
    """
    Impute missing values in a DataFrame using SimpleImputer.

    Parameters:
    - df: pandas DataFrame
      The input DataFrame with missing values.
    - strategy: str, default='most_frequent'
      The imputation strategy. It can be 'constant', 'mean', 'median', or 'most_frequent'.
    - **kwargs: keyword arguments
      Additional arguments that can be passed to SimpleImputer.

    Returns:
    - pandas DataFrame
      A new DataFrame with missing values imputed using the specified strategy.
    """
    imputer = SimpleImputer(strategy=strategy, **kwargs)
    imputed_df = df.copy()
    imputed_df.iloc[:, :] = imputer.fit_transform(imputed_df)
    return imputed_df




def knn_impute_dataframe(df, n_neighbors=2, weights="uniform", **kwargs):
    """
    Impute missing values in a DataFrame using K-Nearest Neighbors (KNN) imputation.

    Parameters:
    - df: pandas DataFrame
      The input DataFrame with missing values.
    - n_neighbors: int, default=2
      The number of neighbors to consider for KNN imputation.
    - weights: str or callable, default="uniform"
      The weight function used for imputation. It can be "uniform" or a callable function.
    - **kwargs: keyword arguments
      Additional arguments that can be passed to KNNImputer.

    Returns:
    - pandas DataFrame
      A new DataFrame with missing values imputed using KNN imputation.
    """
    knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, **kwargs)
    imputed_data = knn_imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
    return imputed_df


def iterative_impute_dataframe(df, **kwargs):
    """
    Impute missing values in a DataFrame using IterativeImputer.

    Parameters:
    - df: pandas DataFrame
      The input DataFrame with missing values.
    - **kwargs: keyword arguments
      Additional arguments that can be passed to IterativeImputer.

    Returns:
    - pandas DataFrame
      A new DataFrame with missing values imputed using IterativeImputer.
    """
    iterative_imputer = IterativeImputer(**kwargs)
    imputed_data = iterative_imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
    return imputed_df



def impute_data_with_mice_and_simple(df, estimator_params={}, mice_params={}, simple_params={}, num_corr_features=10):
    """
    Impute missing values in a DataFrame using a combination of MICE and SimpleImputer.

    Parameters:
    - df: pandas DataFrame
      The input DataFrame with missing values.
    - estimator_params: dict, default={}
      Parameters for the estimator used in MICE imputation (e.g., XGBRegressor parameters).
    - mice_params: dict, default={}
      Parameters for the IterativeImputer used for MICE imputation.
    - simple_params: dict, default={}
      Parameters for the SimpleImputer used for non-correlated features.
    - num_corr_features: int, default=10
      The number of correlated features to use for MICE imputation.

    Returns:
    - pandas DataFrame
      A new DataFrame with missing values imputed using the specified methods.
    """
    X = pd.DataFrame(df)
    #  X = df_mice.drop('y', axis=1)
    #  y = df_mice['y']

    # Select the top k correlated feature names between themselves
    def select_top_k_corr_features(data, k=num_corr_features):
        corr_matrix = data.corr().abs()
        sorted_corr = corr_matrix.unstack().sort_values(ascending=False)
        top_k_features = sorted_corr[sorted_corr < 1].index[:k]
        feature_names = top_k_features.get_level_values(0).unique().tolist()
        return feature_names

    top_k_corr_feature_names = select_top_k_corr_features(X, k=num_corr_features)

    estimator = XGBRegressor(**estimator_params)
    mice_imputer = IterativeImputer(estimator=estimator, random_state=42, **mice_params)

    remaining_feature_names = [col for col in X.columns if col not in top_k_corr_feature_names]
    simple_imputer = SimpleImputer(**simple_params)

    # Use ColumnTransformer to apply the imputers to different sets of columns
    preprocessor = ColumnTransformer(transformers=[
        ('mice', mice_imputer, top_k_corr_feature_names),
        ('simple', simple_imputer, remaining_feature_names)
    ], remainder='passthrough')

    X_imputed = preprocessor.fit_transform(X)

    X_imputed_df = pd.DataFrame(X_imputed, columns=top_k_corr_feature_names + remaining_feature_names)
    #  X_imputed_df['y'] = y
    return X_imputed_df



def impute_data_with_mice_V2(df, estimator_params={}, mice_params={}, simple_params={}, num_corr_features=0.5):
    """
    Impute missing values in a DataFrame using a combination of MICE and SimpleImputer.

    Parameters:
    - df: pandas DataFrame
      The input DataFrame with missing values.
    - estimator_params: dict, default={}
      Parameters for the estimator used in MICE imputation (e.g., XGBRegressor parameters).
    - mice_params: dict, default={}
      Parameters for the IterativeImputer used for MICE imputation.
    - simple_params: dict, default={}
      Parameters for the SimpleImputer used for non-correlated features.
    - num_corr_features: float, default=0.5
      The correlation coefficient to filter features to use with MICE.

    Returns:
    - pandas DataFrame
      A new DataFrame with missing values imputed using the specified methods.
    """
    X = pd.DataFrame(df)
    def select_highly_corr_features(data, corr_coef=0.8):
        corr_matrix = data.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)  # Set diagonal values to 0 to avoid self-correlation

        # Find pairs of highly correlated features
        pairs = np.where(corr_matrix > corr_coef)
        
        # Collect the unique feature names
        unique_feature_names = set()
        for i, j in zip(*pairs):
            unique_feature_names.add(data.columns[i])
            unique_feature_names.add(data.columns[j])
        
        return list(unique_feature_names)



    top_k_corr_feature_names = select_highly_corr_features(X, corr_coef=num_corr_features)
    print(len(top_k_corr_feature_names))

    estimator = XGBRegressor(**estimator_params)
    mice_imputer = IterativeImputer(estimator=estimator, random_state=42, **mice_params)

    remaining_feature_names = [col for col in X.columns if col not in top_k_corr_feature_names]
    simple_imputer = SimpleImputer(**simple_params)

    # Use ColumnTransformer to apply the imputers to different sets of columns
    preprocessor = ColumnTransformer(transformers=[
        ('mice', mice_imputer, top_k_corr_feature_names),
        ('simple', simple_imputer, remaining_feature_names)
    ], remainder='passthrough')

    X_imputed = preprocessor.fit_transform(X)

    X_imputed_df = pd.DataFrame(X_imputed, columns=top_k_corr_feature_names + remaining_feature_names)

    return X_imputed_df
