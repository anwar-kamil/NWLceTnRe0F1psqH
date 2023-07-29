from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split


svc_param_dict = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf']}


def get_test_train_split(dataset, target_index):
    X = dataset.drop(dataset.columns[target_index], axis=1)
    y = pd.DataFrame(dataset.iloc[:, target_index])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    return [X_train, X_test, y_train, y_test]


def default_svc_model():
    return svm.SVC()


def get_grid_search_cv_obj(model, refit=True, verbose=False, n_splits=5, n_repeats=3, random_state=1):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    global svc_param_dict
    grid_search_model = GridSearchCV(model, svc_param_dict, refit=refit, verbose=verbose, cv=cv)
    return grid_search_model


def train_model(model, train_features, train_labels):
    return model.fit(train_features, train_labels)
