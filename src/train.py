import src.build as build
import src.preprocess as preprocess
from inception.aws import bucket_name, s3_file 

def fit():
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess.load(bucket_name, s3_file)
    fit_grid = build.gridsearch()
    fit_grid.fit(X_train, y_train)
    return fit_grid

if __name__ == '__main__':
    fit_grid = fit()
