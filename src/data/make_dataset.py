import pandas as pd
from imblearn.under_sampling import NearMiss


def read_data(input_file):
    """Read data from the input file."""
    data = pd.read_csv(input_file)
    return data


def change_column_name(df, col_index_label):
    column_names = df.columns.tolist()
    for key, value in col_index_label.items():
        df = df.rename(columns={column_names[key]: value})
    return df


def concat_features_target(df_features, df_labels):
    concatenated_df = pd.concat([df_features, df_labels], axis=1)
    return concatenated_df


def undersample_data(df_features, df_labels):
    nm1 = NearMiss(version=1)
    features_resampled, labels_resampled = nm1.fit_resample(df_features, df_labels)
    return concat_features_target(features_resampled, labels_resampled)
