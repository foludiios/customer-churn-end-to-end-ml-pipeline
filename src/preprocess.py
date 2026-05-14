from inception.aws import s3, bucket_name, s3_file
import pandas as pd
from sklearn.model_selection import train_test_split
from io import StringIO
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler

def load(bucket_name_, s3_file_path): # folder + file (full ingested data excluding bucket name, that's clearly a separate argument)
    obj = s3.Object(bucket_name_, s3_file_path)
    data = obj.get()['Body'].read().decode('utf-8')  # read content as string 
    df = pd.read_csv(StringIO(data))

    X, y = df[df.columns[:-1]], df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(0.2/0.8), random_state=69)
    return X_train, X_val, X_test, y_train, y_val, y_test

def prp_scale():
    num_cols = ['Call Failure', 'Subscription Length', 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers',
     'Age', 'Customer Value']
    ord_cols = ['Charge Amount', 'Age Group']

    processor2 = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_cols),
            ("ord", StandardScaler(), ord_cols)],
            remainder = 'passthrough')
    return processor2
