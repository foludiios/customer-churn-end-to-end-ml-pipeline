import src.preprocess as preprocess
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from inception.aws import bucket_name, s3_file
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_model(layers_config=[128, 64, 32], activations=['relu', 'relu', 'sigmoid'], learning_rate=0.001):
    """ layers_config: list of integers for number of neurons per hidden layer activations: list of activation functions per layer (strings 
    or Keras layers)"""

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess.load(bucket_name, s3_file)
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))

    for neurons, act in zip(layers_config, activations):
        model.add(Dense(neurons))
        if act != 'leaky':
            model.add(tf.keras.layers.Activation(act))
        else:
            model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

model = KerasClassifier(model=build_model, epochs=100, batch_size=100, class_weight={0: 1.5, 1: 5.5}, verbose=1)

pipeline = Pipeline([
    ('scale', preprocess.prp_scale()),
    ('model', model)
])

def gridsearch():
    param_grid = {
        'model__model__layers_config': [ # pipeline step name __ parameter in that step
            [128, 64, 32],
            [64, 32, 16],
            [128, 128, 64]
        ],
        'model__model__activations': [
            ['relu', 'relu', 'relu'],
            ['leaky', 'leaky', 'relu'],
            ['relu', 'leaky', 'leaky']
        ],
        'model__model__learning_rate': [0.001, 0.0005],
        'model__batch_size': [100, 200]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='precision', verbose=2)
    return grid