import requests
import src.preprocess as preprocess
from inception.aws import bucket_name, s3_file
import pandas as pd

url = "http://127.0.0.1:8000/predict"

def req(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="records")
    response = requests.post(
        url, # after running api's xyz:xyz --reload 
        json={"data": data}
    )
    print(response.json())


_, X_val, X_test, _, y_val, y_test = preprocess.load(bucket_name, s3_file)
req(X_test)
