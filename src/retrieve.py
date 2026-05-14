from inception.aws import s3, bucket_name
from src.evaluate import folder
import tensorflow as tf
import joblib
import json

retr_model = None
retr_processor = None

def retrieval():
    global retr_model, retr_processor

    if retr_processor is None or retr_model is None:
        s3.Bucket(bucket_name).download_file(folder + "preprocess.pkl", "preprocess.pkl")
        s3.Bucket(bucket_name).download_file(folder + "model.keras", "model.keras")
        s3.Bucket(bucket_name).download_file(folder + "metadata.json", "metadata.json")
    
        retr_model = tf.keras.models.load_model("model.keras")
        retr_processor = joblib.load("preprocess.pkl")

    try:
        with open("metadata.json", "r") as f: 
            metadata = json.load(f)
        threshold = metadata.get("threshold", 0.5)
    except Exception:
        threshold = 0.5

    return retr_processor, retr_model, threshold
