from src.config import BASE_PATH

import pandas as pd
import joblib as jb


def get_data(outcome_folder, file_dir=BASE_PATH / "data" / "processed/"):
    """
    For a given outcome, get X/y train, validation, and testing data
    """
    file_path = file_dir / outcome_folder
    return {
        "X_train": pd.read_parquet(file_path / "X_train.parquet"),
        "y_train": pd.read_excel(file_path / "y_train.xlsx", index_col=0),
        "X_val": pd.read_parquet(file_path / "X_val.parquet"),
        "y_val": pd.read_excel(file_path / "y_val.xlsx", index_col=0),
        "X_test": pd.read_parquet(file_path / "X_test.parquet"),
        "y_test": pd.read_excel(file_path / "y_test.xlsx", index_col=0),
    }


def get_models(model_prefix_list, outcome, file_dir=BASE_PATH / "models"):
    """
    For a given outcome, get all models that predict that outcome
    """
    model_dict = {}
    for model_name in model_prefix_list:
        model = jb.load(file_dir / outcome / f"{model_name}.joblib")
        model_dict[model_name] = model
    return model_dict


def get_feature_lists(df):
    """
    Classify all features/columns in a dataframe as one of:
        binary, nominal, ordinal, categorical using the number of unique values
        in that column
    """
    binary_cols = []
    nominal_cols = []
    ordinal_cols = ["ASACLAS"]
    numerical_cols = []
    for col in df.columns:
        if col in ordinal_cols:
            continue
        val_counts = df[col].value_counts()
        if len(val_counts) == 2:
            binary_cols.append(col)
        elif len(val_counts) <= 20:
            nominal_cols.append(col)
        else:
            numerical_cols.append(col)
    return {
        "binary_cols": list(binary_cols),
        "numerical_cols": list(numerical_cols),
        "nominal_cols": list(nominal_cols),
        "ordinal_cols": list(ordinal_cols),
    }
