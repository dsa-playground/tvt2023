# Imports
import os
import pandas as pd
from pyaml_env import parse_config

from scripts.preprocess.preprocess_load import load_csv, return_config
from scripts.preprocess.preprocess_clean import rename_cols, change_order_cols, add_sum_column, drop_cols, \
    replace_values_cols, floor_values_cols, fillna_cols, dropna_rows, create_index
from scripts.preprocess.preprocess_collect import add_multiple_records, transform_multiplechoice_anwser, \
    transform_multi_records_to_df

#Load settings
config_path="../settings.yml"

_config = parse_config(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))

def zie_settings(config=_config):
    """Returns the json with settings as noted in settings.yml.

    Parameters
    ----------
    config : dict, optional
        Dictionary with all settings, by default _config (which is loaded in this script).

    Returns
    -------
    dict
        Dictionary format of the settings.
    """
    return return_config(config)


def laden_data(config=_config):
    """Loads the train and test data of the Titanic Dataset stored in the
    folder 'data'. 

    Parameters
    ----------
    config : dict, optional
        Dictionary with settings, refering to the path of the csv's, by default _config

    Returns
    ----------
    pd.DataFrame(s)
        Two pd.DataFrames with the train and test dataset. 
    """
    df_train = load_csv(path = config['preprocess']['data']['train'])
    df_test = load_csv(path = config['preprocess']['data']['test'])

    return df_train, df_test

def opschonen_dataframe(df, config=_config):
    """Cleans values of DataFrame to make it more readable for Dutch users.

    Steps taken in this function:
    1. Rename columns
    2. Add summerized column(s)
    3. Change order of columns
    4. Replace values of certain columns

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with number of columns
    config : dict, optional
        Dictionary with multiple settings, by default _config

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with readable column values and columnnames for Dutch audience.
    """
    df_clean = df.copy()
    df_clean = rename_cols(df=df_clean, dict_renamed_columns=config['preprocess']['clean']['rename'])
    for item in _config['preprocess']['clean']['add_sum_columns']['items'].keys():
        df_clean = add_sum_column(df=df_clean, 
                                  variable_name=_config['preprocess']['clean']['add_sum_columns']['items'][item]['colname'], 
                                  list_columns_to_sum=_config['preprocess']['clean']['add_sum_columns']['items'][item]['columns'])
    df_clean = (df_clean.pipe(change_order_cols, list_order_cols=config['preprocess']['clean']['order'])
            .pipe(replace_values_cols, dict_replacing=config['preprocess']['clean']['replace'])
            )
    
    return df_clean


def numeriek_maken_dataframe(df, config=_config):
    """Transforms all columns to numeric values.

    Steps taken in this function:
    1. Drop certain columns
    2. Fillna for certain columns
    3. Drop rows with NaNs
    4. Floor floats & astype int
    5. Replace values of certain columns
    6. Create index (for certain [non-numeric] columns)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with number of columns
    config : dict, optional
        Dictionary with multiple settings, by default _config

    Returns
    -------
    pd.DataFrame
        Numeric DataFrame with suitable for modeling.
    """
    df_num = df.copy()
    df_num = (df_num.pipe(drop_cols, list_drop_cols=config['preprocess']['clean']['drop_cols'])
            .pipe(fillna_cols, dict_fillna_strategy=config['preprocess']['clean']['fillna_strategy'])
            .pipe(dropna_rows)
            .pipe(floor_values_cols, list_rounding_cols=config['preprocess']['clean']['floor'])
            .pipe(replace_values_cols, dict_replacing=config['preprocess']['clean']['label_encode'])
            .pipe(create_index, list_index_cols=config['preprocess']['clean']['index_cols'])
            )
    
    return df_num


def opschonen_data(df_train, df_test, config=_config):
    """Clean train and test DataFrame of Titanic datasets in simular way.

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame of train dataset.
    df_test : pd.DataFrame
        DataFrame of test dataset.
    config : dict, optional
        Dictionary with multiple settings, by default _config

    Returns
    -------
    pd.DataFrame(s)
        Cleaned DataFrames of train and test.
    """
    df_train_cleaned = opschonen_dataframe(df=df_train, config=config)
    df_test_cleaned = opschonen_dataframe(df=df_test, config=config)
    
    return df_train_cleaned, df_test_cleaned


def numeriek_maken_data(df_train_clean, df_test_clean, config=_config):
    """Make train and test DataFrame of Titanic datasets numeric.

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame of train dataset.
    df_test : pd.DataFrame
        DataFrame of test dataset.
    config : dict, optional
        Dictionary with multiple settings, by default _config

    Returns
    -------
    pd.DataFrame(s)
        Numeric DataFrames of train and test for modeling.
    """
    df_train_num = numeriek_maken_dataframe(df=df_train_clean, config=config)
    df_test_num = numeriek_maken_dataframe(df=df_test_clean, config=config)
    
    return df_train_num, df_test_num

def voeg_passagiers_toe(df_train, df_test, config=_config):
    """Add new records to df_train (for EDA purposes) and df_test for predicting new passengers.

    Steps taken in this function:
    1. Add multiple records by collecting anwsers.
    2. Tranform multiple choice anwser to multiple anwsers.
    3. Transform to DataFrame.
    4. Add summerized column (like in cleaning step).
    5. Replace values of certain columns
    6. Create index (for certain [non-numeric] columns)
    7. Concat with original train and test datasets.

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame of train dataset after making DataFrame numeric.
    df_test : pd.DataFrame
        DataFrame of test dataset after making DataFrame numeric.
    config : _type_, optional
        _description_, by default _config

    Returns
    -------
    pd.DataFrame(s)
        Numeric DataFrames of train and test for modeling.
    """
    df_train_added = df_train.copy()
    df_test_added = df_test.copy()

    list_workshop_passagiers = add_multiple_records(
        dict_with_items_to_collect=_config['preprocess']['collect']['items_to_collect'])
    list_workshop_passagiers_updated=transform_multiplechoice_anwser(
        list_with_dicts=list_workshop_passagiers,
        dict_with_multiplechoice_anwsers=_config['preprocess']['collect']['transform_multi'])
    df_workshop_passagiers = transform_multi_records_to_df(list_with_all_new_records=list_workshop_passagiers_updated, 
                                                           list_with_drop_columns=_config['preprocess']['collect']['drop_cols'])
    for item in _config['preprocess']['clean']['add_sum_columns']['items'].keys():
        df_workshop_passagiers = add_sum_column(df=df_workshop_passagiers, 
                                  variable_name=_config['preprocess']['clean']['add_sum_columns']['items'][item]['colname'], 
                                  list_columns_to_sum=_config['preprocess']['clean']['add_sum_columns']['items'][item]['columns'])
    df_workshop_passagiers = (df_workshop_passagiers.pipe(replace_values_cols, dict_replacing=config['preprocess']['clean']['label_encode'])
                              .pipe(create_index, list_index_cols=config['preprocess']['clean']['index_cols']))

    df_train_added = pd.concat([df_train_added, df_workshop_passagiers], join="outer")
    df_test_added = pd.concat([df_test_added, df_workshop_passagiers], join="outer")

    return df_train_added, df_test_added