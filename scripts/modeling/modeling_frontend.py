# Imports
import os
import pandas as pd
from pyaml_env import parse_config

from scripts.modeling.modeling import create_knn_model, save_model2pickle, load_model_from_pickle

#Load settings
config_path="../settings.yml"

_config = parse_config(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))


def train_and_save_model(df, config=_config):
    # Exclude possible extra entries
    df_train = df.iloc[df.index.get_level_values('Passagier_Id')<10_000].copy()
    # Train model
    model = create_knn_model(df=df_train, y_column=config['modeling']['y_variable'])
    # Save model
    save_model2pickle(model=model, default_filename=config['modeling']['filename'])
    

def voorspelling_genereren(X, config=_config):
    # load model
    model = load_model_from_pickle(default_filename=config['modeling']['filename'])
    # predict
    yhat=model.predict(X)
    # combine results in one DataFrame
    X[config['modeling']['y_variable']]=yhat
    return X

def evalueer():
    pass