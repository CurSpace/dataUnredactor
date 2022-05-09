import project3
import pytest
import warnings
warnings.filterwarnings("ignore")

@pytest.fixture()
def test_data():
    data = "unredactor.tsv"
    return data

def test_splitDdata(test_data):
    train,valid,test = project3.splitData(test_data, flag = 1)
    assert train.shape == (2343, 4)
    assert valid.shape == (1400, 4)
    assert test.shape == (503, 4)

def test_extractFeatures(test_data):
    train,valid,test = project3.splitData(test_data,flag = 1)
    X_train,X_valid,X_test,y_train,y_valid,y_test = project3.extractFeatures(train,valid,test)
    assert X_train.shape == (2343, 7463)
    assert X_valid.shape == (1400, 7463)
    assert X_test.shape == (503, 7463)
    assert y_train.shape == (2343,)
    assert y_valid.shape == (1400,)
    assert y_test.shape == (503,)

def test_training(test_data):
    train,valid,test = project3.splitData(test_data,flag = 1)
    X_train,X_valid,X_test,y_train,y_valid,y_test = project3.extractFeatures(train,valid,test)
    prediction = project3.training(X_train,y_train,X_test)
    assert len(prediction)

def test_metrics(test_data):
    train,valid,test = project3.splitData(test_data,flag = 1)
    X_train,X_valid,X_test,y_train,y_valid,y_test = project3.extractFeatures(train,valid,test)
    predictions = project3.training(X_train,y_train,X_test)
    precision,recall,F1Score = project3.metrics(y_test,predictions)
    assert precision > 0 and precision < 1
    assert recall > 0 and recall  < 1
    assert F1Score > 0 and F1Score < 1
