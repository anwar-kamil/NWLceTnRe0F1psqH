from sklearn.metrics import f1_score

import data.make_dataset as md
import features.build_features as bf
import visualizations.visualize as vis
import helpers.helper as util
import models.predict_model as pm
import models.train_model as tm
import numpy as np

file_path = "../data/raw/term-deposit-marketing-2020.csv"


def main():
    # Step 1: Make dataset
    global file_path
    dataset = md.read_data(file_path)

    dataset = md.change_column_name(dataset, {4: 'has_credit_default', 6: 'has_housing_loan', 7: 'has_personal_loan',
                                             8: 'contact_mode', 12: 'num_of_contacts'})

    # Step 2: Generate visualization report
    #TO-DO vis.auto_eda(dataset, "Term Deposit EDA Report")

    # Step 3: Build features
    # Dropping irrelevant data: day and month
    dataset_col_dropped = bf.drop_columns_by_index(dataset, [9, 10])
    # Scaling age, balance, duration, num_contacts
    dataset_scaled = bf.scale_numeric_features(dataset_col_dropped,
                                               [0, 5, 9, 10])

    col_names = ['job', 'marital', 'education', 'has_credit_default', 'has_housing_loan', 'has_personal_loan',
                 'contact_mode', 'y']
    dataset_encoded = bf.one_hot_encode_features(dataset_scaled, col_names) # One-hot encoding

    binary_cols = ['y_no', 'has_personal_loan_no', 'has_housing_loan_no', 'has_credit_default_no']
    # Get column indices
    binary_col_indices = util.get_col_indices(dataset_encoded, binary_cols)
    # Drop columns which are binary
    engineered_features = bf.drop_columns_by_index(dataset_encoded, binary_col_indices)

    features, labels = bf.get_features_labels(engineered_features, 29)
    label_array = np.array(labels).ravel()
    # Feature selection
    feature_selected_dataset = bf.select_features(features.values, label_array)

    #Undersampling using NearMiss strategy
    transformed_dataset = md.undersample_data(feature_selected_dataset, labels)

    # Step 3: Train model and make predictions
    X_train, X_test, y_train, y_test = tm.get_test_train_split(transformed_dataset, 11)
    y_train_array = np.array(y_train)
    model = tm.default_svc_model()
    optimized_model = tm.get_grid_search_cv_obj(model)

    trained_model = tm.train_model(optimized_model, X_train, y_train_array.ravel())
    label_predictions = pm.test_model(trained_model, X_test)
    score = f1_score(y_test, label_predictions, average="weighted")
    print('Model F1 score: {0:0.3f}'.format(score))


if __name__ == "__main__":
    main()
