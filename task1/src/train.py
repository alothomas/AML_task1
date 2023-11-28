from task1.pre.preprocessing import DataPreprocessor
from task1.src.model import MODEL_TYPES, r2_score
from task1.utils import flatten_dict
import yaml
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split

file_path_parser = argparse.ArgumentParser()

file_path_parser.add_argument("--config_path")


def initialize_model(model_type, model_config):
    """
    Initialize a model based on the given type and configuration.

    Parameters
    ----------
    model_type : str
        Type of the model to be initialized.
    model_config : dict
        Configuration parameters for the model.

    Returns
    -------
    model : object
        An instance of the specified model type.
    """
    model = MODEL_TYPES[model_type](**model_config)
    return model


def load_data(data_paths):
    """
    Load features and labels from the provided data paths.

    Parameters
    ----------
    data_paths : dict
        Dictionary containing paths to the features and labels.
        Should have keys "features" and "label".

    Returns
    -------
    X : pd.DataFrame
        Feature data with "id" as index.
    y : pd.DataFrame
        Label data with "id" as index.
    """
    X = pd.read_csv(data_paths["features"]).set_index("id")
    if os.path.exists(data_paths["label"]):
        y = pd.read_csv(data_paths["label"]).set_index("id")
    else:
        y = None
    return X, y


def preprocess_data(X_train_df, y, X_valid_df, preprocessing_config):
    data_preprocessor = DataPreprocessor(
        preprocessing_config
    )
    """
    Preprocess feature and label data.

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature data.
    y : np.ndarray
        Label data.
    preprocessing_config : dict
        Configuration for data preprocessing.
    selected_features : list, optional
        Indices of selected features, by default None.

    Returns
    -------
    X_train_mat : np.ndarray
        Processed feature data of train split in numpy array format.
    X_valid_mat : np.ndarray
        Processed feature data of valid split in numpy array format.
    """
    X_train_mat, X_valid_mat, y_train_mat = data_preprocessor.preprocess(X_train_df, y, X_valid_df)
    return X_train_mat, X_valid_mat, y_train_mat


def train_and_get_prediction(
    model, X_train_df, y_train_df, X_valid_df, y_valid_df, preprocessing_config
):
    """
    Train and output prediction.

    Parameters
    ----------
    model : object
        Model to be trained and evaluated.
    X_train_df : pd.DataFrame
        Training feature data.
    y_train_df : pd.DataFrame
        Training label data.
    X_valid_df : pd.DataFrame
        Validation feature data.
    y_valid_df : pd.DataFrame
        Validation label data.
    preprocessing_config : dict
        Configuration for data preprocessing.

    Returns
    -------
    y_valid : Optional[np.ndarray]
        Valid label if exists.
    y_pred : np.ndarray
        Prediction for the labels.
    """
    y_train_mat = y_train_df.y.to_numpy()
    if y_valid_df is not None:
        y_valid_mat = y_valid_df.y.to_numpy()
    else:
        y_valid_mat = None
    X_train_mat, X_valid_mat, y_train_mat = preprocess_data(
        X_train_df, y_train_mat, X_valid_df, preprocessing_config
    )
    model.fit(X_train_mat, y_train_mat)

    y_pred = model.predict(X_valid_mat)
    return y_valid_mat, y_pred

def save_train_results(config, score):
    curr_config_and_results_df = pd.DataFrame.from_dict(
        {**flatten_dict(config), **{'train_results': score}},
        orient='index'
    ).T
    if os.path.exists(config['train']["results_path"]):
        config_and_results_df = pd.read_csv(config['train']["results_path"])
        common_config_keys = config_and_results_df.columns.intersection(
            curr_config_and_results_df.columns
        ).tolist()
        curr_config_and_results_df = pd.merge(
            config_and_results_df,
            curr_config_and_results_df,
            how="outer",
            on=common_config_keys,
        )

    curr_config_and_results_df.to_csv(config['train']["results_path"], index=False)

def main(config):
    train_config = config["train"]
    model_config = train_config["model_config"]
    model_type = train_config["model_type"]
    model = initialize_model(model_type, model_config)

    data_paths = train_config["data_paths"]
    X_train_df, y_train_df = load_data(data_paths)

    preprocessing_config = train_config["preprocessing"]
    if train_config["cross_validation"] > 1:
        kf = KFold(
            n_splits=train_config["cross_validation"], shuffle=True, random_state=42
        )
        scores = []
        for train_idx, valid_idx in kf.split(X_train_df):
            subset_X_train_df, subset_y_train_df = (
                X_train_df.iloc[train_idx],
                y_train_df.iloc[train_idx],
            )
            subset_X_valid_df, subset_y_valid_df = (
                X_train_df.iloc[valid_idx],
                y_train_df.iloc[valid_idx],
            )
            subset_y_valid, subset_y_pred = train_and_get_prediction(
                model,
                subset_X_train_df,
                subset_y_train_df,
                subset_X_valid_df,
                subset_y_valid_df,
                preprocessing_config,
            )
            valid_score = r2_score(subset_y_valid, subset_y_pred)
            scores.append(valid_score)
        score = np.array(scores).mean()
    else:
        X_train_df, X_valid_df, y_train_df, y_valid_df = train_test_split(
            X_train_df, y_train_df, test_size=0.33, random_state=42
        )
        y_valid, y_pred = train_and_get_prediction(
            model, X_train_df, y_train_df, X_valid_df, y_valid_df, preprocessing_config
        )
        score = r2_score(y_valid, y_pred)

    if config['evaluate']['run']:
        X_valid_df, y_valid_df = load_data(config['evaluate']['data_paths'])
        _, y_pred = train_and_get_prediction(model, X_train_df, y_train_df, X_valid_df, y_valid_df, preprocessing_config)
        y_pred_path = config['evaluate']['data_paths']['label']
        pd.DataFrame(y_pred).reset_index().rename(columns={'index': 'id', 0: 'y'}).astype(float).to_csv(y_pred_path, index=False)

    save_train_results(config, score)
    return score


if __name__ == "__main__":
    args = file_path_parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    main(config)