from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


def drop_columns_by_index(df, indices):
    df = df.drop(df.columns[indices], axis=1)
    return df


def scale_numeric_features(df, col_indices):
    scaler = MinMaxScaler()
    df.iloc[:, col_indices] = scaler.fit_transform(df.iloc[:, col_indices])
    return df


def one_hot_encode_features(df, col_names):
    df_encoded = pd.get_dummies(df, columns = col_names)
    return df_encoded


def get_features_labels(df, label_index):
    X = df.drop(df.columns[label_index], axis=1)
    y = pd.DataFrame(df.iloc[:, label_index])
    return [X, y]


def select_features(features, target):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, target)
    model = SelectFromModel(lsvc, prefit=True)
    features_new = model.transform(features)
    df_feature_selected = pd.DataFrame(features_new)
    return df_feature_selected
