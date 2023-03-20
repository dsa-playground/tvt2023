# Imports
import os
import pandas as pd
from pyaml_env import parse_config

from scripts.modeling.modeling import knn_experiment, knn_with_specific_neighbour, create_knn_model, save_model2pickle, load_model_from_pickle

#Load settings
config_path="../settings.yml"

_config = parse_config(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))

def experimenteer_met_aantal_buren(df, ondergrens=1, bovengrens=20, config=_config):
    # Exclude possible extra entries
    df_train = df.iloc[df.index.get_level_values('Passagier_Id')<10_000].copy()
    df_experiment = knn_experiment(df=df_train, 
                                   y_column=config['modeling']['y_variable'],  
                                   lower_boundary_range=ondergrens, 
                                   upper_boundary_range=bovengrens)
    return df_experiment


def verdieping_specifiek_model(df, aantal_buren, config=_config):
    # Exclude possible extra entries
    df_train = df.iloc[df.index.get_level_values('Passagier_Id')<10_000].copy()
    # Train model and get confusion_matrix
    model, df_confusion_matrix = knn_with_specific_neighbour(
        df=df_train, 
        k=aantal_buren,
        y_column=config['modeling']['y_variable'])
    # Change index and column to logical names for this use case
    multiindex_values = [('Werkelijk','Overleden'),('Werkelijk','Overleefd')]
    columns_values = [('Voorspeld','Overleden'),('Voorspeld','Overleefd')]
    df_confusion_matrix = df_confusion_matrix.set_index(pd.MultiIndex.from_tuples(multiindex_values))
    df_confusion_matrix.columns = pd.MultiIndex.from_tuples(columns_values)
    # Save model
    save_model2pickle(model=model, default_filename=config['modeling']['filename'])
    return df_confusion_matrix


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