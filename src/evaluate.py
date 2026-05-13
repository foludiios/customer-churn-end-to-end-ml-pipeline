import src.train as train
import joblib
from inception.aws import s3, bucket_name, s3_file
import src.preprocess as preprocess
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json

folder = "Model 1/"

def get_current_best_score():
    try:
        obj = s3.Object(bucket_name, folder + "metadata.json")
        metadata = json.loads(obj.get()['Body'].read().decode('utf-8'))
        return metadata.get("score", 0)
    except Exception:
        return 0
current_best_score = get_current_best_score()

# recall, best workflow is create model, choose threshold, get best results, decide if said best 
# results are good enough

def best_est():
    fit_grid = train.fit()

    best_model = fit_grid.best_estimator_
    best_params = fit_grid.best_params_
    #best_score = fit_grid.best_score_
    
    _, X_val, _, _, y_val, _ = preprocess.load(bucket_name, s3_file)
    val_preds = best_model.predict(X_val).flatten() # model was trained in pipeline combined with preprocessing so no need to preprocess xval separately

    return best_model, best_params, y_val, val_preds # find best model

def best_threshold(y_valid, valid_preds, metrics=['recall'], weights=[1]):

    thresholds = np.linspace(0.01, 0.99, 100)
    metric_map = {
        'recall': recall_score,
        'precision': precision_score,
        'f1': f1_score
    }
    for metric in metrics:
        if metric not in metric_map:
            raise ValueError(f"Unsupported metric: {metric}")

    if isinstance(weights, list):
        if len(weights) != len(metrics):
            raise ValueError("metrics and weights must have same length")
        weight = dict(zip(metrics, weights))
    else:
        weight = weights

    best_thresh = 0
    best_score = -1
    best_metric_scores = {}

    for t in thresholds:
        preds = (valid_preds > t).astype(int)
        scores = {}

        for metric in metrics:
            metric_func = metric_map[metric]
            scores[metric] = metric_func(y_valid, preds)
        combined_score = sum(scores[m] * weight[m] for m in metrics)

        if combined_score > best_score:
            best_score = combined_score
            best_thresh = t
            best_metric_scores = scores.copy()

    return best_score, best_thresh, best_metric_scores # find it's best score 
    
def save_model(best_model, best_params, best_score, best_metric_scores, best_thresh):
    #actual model and preprocessor are saved separately (for 'dependency issues') and this is why tha api calls both preprocessing and model separately instead of just model as was done on x_val
    keras_model = best_model.named_steps['model'].model_  ##named step from pipeline(), extrastep due to keras classifier retrieval
    preprocessor = best_model.named_steps['scale']
    metadata = {
        "threshold": float(best_thresh),
        "score": float(best_score),
        "params": best_params,
        "metric_scores": best_metric_scores
    }

    with open("metadata.json", "w") as f:
        json.dump(metadata, f)
    joblib.dump(preprocessor, "preprocess.pkl")
    keras_model.save("model.keras")
    s3.Bucket(bucket_name).upload_file("metadata.json", folder + "metadata.json")
    s3.Bucket(bucket_name).upload_file("preprocess.pkl", folder + "preprocess.pkl")
    s3.Bucket(bucket_name).upload_file("model.keras", folder + "model.keras")

def main():
    best_model, best_params, y_val, val_preds = best_est()
    best_score, best_thresh, best_metric_scores = best_threshold(y_val, val_preds, metrics=['precision', 'recall'], weights=[.4, .6])
    if best_score > current_best_score:
        save_model(best_model, best_params, best_score, best_metric_scores, best_thresh)

if __name__ == '__main__':
    main()