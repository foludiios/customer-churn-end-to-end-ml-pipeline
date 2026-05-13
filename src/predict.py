import pandas as pd
from src.retrieve import retrieval
from sklearn.metrics import classification_report
from inception.aws import s3, bucket_name, s3_file
import src.preprocess as preprocess


def make_prediction(datapoints):
    retr_processor, retr_model, threshold = retrieval()

    feature_order = ["Call Failure", "Complaints", "Subscription Length", "Charge Amount", "Seconds of Use", "Frequency of use", "Frequency of SMS", 
    "Distinct Called Numbers", "Age Group", "Tariff Plan", "Status", "Age", "Customer Value"] # because i want to accept list

    if isinstance(datapoints, pd.DataFrame):
        df = datapoints.copy()
    elif isinstance(datapoints, list):
        if len(datapoints) == len(feature_order) and not isinstance(datapoints[0], list): #single list of single datapoint
            df = pd.DataFrame([datapoints], columns=feature_order)
        else: # single list of multiple lists each containing a datapoint
            df = pd.DataFrame(datapoints, columns=feature_order)
    elif isinstance(datapoints, dict):
        df = pd.DataFrame([datapoints])
    else:
        raise ValueError("Input must be list or dict")

    processed = retr_processor.transform(df)
    prediction = retr_model.predict(processed)
    predictions = (prediction.flatten() > threshold).astype(int)

    return predictions.tolist()


def test():
    _, X_val, X_test, _, y_val, y_test = preprocess.load(bucket_name, s3_file)
    print(make_prediction(X_test))
    print(y_test)
    print(classification_report(y_test, make_prediction(X_test)))

