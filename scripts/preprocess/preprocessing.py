import os
import pandas as pd
from pyaml_env import parse_config
config_path="../settings.yml"

_config = parse_config(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))


def return_config(config=_config):
    """Returns config dictionary to view

    Parameters
    ----------
    config : dict()
        Dictionary with setting parameters from settings.yml.

    Returns
    -------
    dict()
        Dictionary with setting parameters from settings.yml.
    """
    return config


def load_csv(source, config):
    """Import the data from csv into a Pandas DataFrame

    Parameters
    ----------
    source : str
        Text of source, i.e. "train" of "test" data.
    config : dict()
        Dictionary with setting parameters from settings.yml.

    Returns
    ----------
    pd.DataFrame
        DataFrame with train/test data loaded
    """
    return pd.read_csv(config["preprocess"]["data"][source])
    

def laden_data(config=_config):
    """_summary_

    Parameters
    ----------
    config : _type_, optional
        _description_, by default _config

    Returns
    ----------
    pd.DataFrame(s)
        Two pd.DataFrames with the train and test dataset. 
    """
    df_train = load_csv(source="train", config=_config).rename(
        columns=_config["preprocess"]["data"]["rename"]).sort_index(axis=1)
    df_test = load_csv(source="test", config=_config).rename(
        columns=_config["preprocess"]["data"]["rename"]).sort_index(axis=1)

    return df_train, df_test


def give_name(config=_config):
    x = input('Enter your name:')
    return x


def give_input(config=_config):
    name = input('Vul hier je naam in:')
    age = input('Vul hier je leeftijd in:')
    sex = input('Vul hier je geslacht in:')
    kids = input('Vul hier het aantal kinderen in welke mee op reis gaan:')
    family = input('Vul hier het aantal familieleden in welke mee op reis gaan:')
    return [name, age, sex, kids, family]
    

# if __name__ == "__main__":
#     # Load data
#     df_train = import_data(source="train", config=_config)
#     df_train = import_data(source="train", config=_config)
