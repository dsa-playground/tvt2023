# Imports
import os
import pandas as pd
from pyaml_env import parse_config

from scripts.modeling.modeling import knn_experiment, knn_with_specific_neighbor, create_knn_model, save_model2pickle, load_model_from_pickle

#Load settings
config_path="../settings.yml"

_config = parse_config(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))

def experimenteer_met_aantal_buren(df, ondergrens=1, bovengrens=20, config=_config):
    """Experiment with multiple neighbors in a range to see the accuracy score of multiple KNN models.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split into train and test dataset
    ondergrens : int, optional
        Integer to define lower boundary of range for number of neighbors, by default 1
    bovengrens : int, optional
        Integer to define upper boundary of range for number of neighbors, by default 20
    config : dict, optional
        Dictionary with all settings, by default _config (which is loaded in this script).

    Returns
    -------
    pd.DataFrame
        DataFrame with the columns k (for neighbors) and accuracy score for the different KNN models.
    """
    # Exclude possible extra entries
    df_train = df.iloc[df.index.get_level_values('Passagier_Id')<10_000].copy()
    df_experiment = knn_experiment(df=df_train, 
                                   y_column=config['modeling']['y_variable'],  
                                   lower_boundary_range=ondergrens, 
                                   upper_boundary_range=bovengrens)
    return df_experiment


def verdieping_specifiek_model(df, aantal_buren, config=_config):
    """Trains and saves specific model and returns confusion matrix

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split into train and test dataset.
    aantal_buren : int
        Number of neighbors to use by default for kneighbors queries.
    config : dict, optional
        Dictionary with all settings, by default _config (which is loaded in this script).

    Returns
    -------
    pd.DataFrame
        Confusion matrix as a DataFrame.
    Note: saves KNN model in folder 'models'
    """
    # Exclude possible extra entries
    df_train = df.iloc[df.index.get_level_values('Passagier_Id')<10_000].copy()
    # Train model and get confusion_matrix
    model, df_confusion_matrix = knn_with_specific_neighbor(
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
    """Trains and saves KNN model in the folder 'models'.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split into train and test dataset.
    config : dict, optional
        Dictionary with all settings, by default _config (which is loaded in this script).
    """
    # Exclude possible extra entries
    df_train = df.iloc[df.index.get_level_values('Passagier_Id')<10_000].copy()
    # Train model
    model = create_knn_model(df=df_train, y_column=config['modeling']['y_variable'])
    # Save model
    save_model2pickle(model=model, default_filename=config['modeling']['filename'])
    

def voorspelling_genereren(X, config=_config):
    """Loads a model and predicts the target variable.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with same features as trained model.
    config : dict, optional
        Dictionary with all settings, by default _config (which is loaded in this script).

    Returns
    -------
    pd.DataFrame
        DataFrame with all features and target variable.
    """
    # load model
    model = load_model_from_pickle(default_filename=config['modeling']['filename'])
    # predict
    yhat=model.predict(X)
    # combine results in one DataFrame
    X[config['modeling']['y_variable']]=yhat
    return X