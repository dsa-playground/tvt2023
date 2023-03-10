import pandas as pd


def return_config(config):
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


# def load_csv(source, config):
#     """Import the data from csv into a Pandas DataFrame

#     Parameters
#     ----------
#     source : str
#         Text of source, i.e. "train" of "test" data.
#     config : dict()
#         Dictionary with setting parameters from settings.yml.

#     Returns
#     ----------
#     pd.DataFrame
#         DataFrame with train/test data loaded
#     """
#     return pd.read_csv(config["preprocess"]["data"][source])

def load_csv(path):
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
    return pd.read_csv(path)