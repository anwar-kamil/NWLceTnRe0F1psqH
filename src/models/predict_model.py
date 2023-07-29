def test_model(model, test_features):
    predictions = model.predict(test_features)
    return predictions

