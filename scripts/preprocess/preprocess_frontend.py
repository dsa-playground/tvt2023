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
    return return_config(config)


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
    df_train = load_csv(path = config['preprocess']['data']['train'])
    df_test = load_csv(path = config['preprocess']['data']['test'])
    # df_train = load_csv(source="train", config=_config).rename(
    #     columns=_config["preprocess"]["data"]["rename"]).sort_index(axis=1)
    # df_test = load_csv(source="test", config=_config).rename(
    #     columns=_config["preprocess"]["data"]["rename"]).sort_index(axis=1)

    return df_train, df_test

def opschonen_dataframe(df, config=_config):
    #steps to take:
    # 1. rename columns
    # 2. Change order of columns
    # 3. Drop certain columns
    # 4. Replace values of certain columns
    # 5. Fillna for certain columns
    # 6. Drop rows with NaNs
    # 7. Floor floats & astype int
    # 8. Create index (for certain [non-numeric] columns)

    df_clean = df.copy()
    df_clean = (df_clean.pipe(rename_cols, dict_renamed_columns=config['preprocess']['clean']['rename'])
            .pipe(change_order_cols, list_order_cols=config['preprocess']['clean']['order'])
            .pipe(replace_values_cols, dict_replacing=config['preprocess']['clean']['replace'])
            )
    for item in _config['preprocess']['clean']['add_sum_columns']['items'].keys():
        df_clean = add_sum_column(df=df_clean, 
                                  variable_name=_config['preprocess']['clean']['add_sum_columns']['items'][item]['colname'], 
                                  list_columns_to_sum=_config['preprocess']['clean']['add_sum_columns']['items'][item]['columns'])
    
    return df_clean


def numeriek_maken_dataframe(df, config=_config):
    #steps to take:
    # 1. Replace str values by numeric values (label encoding by controled replacing values)

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

    df_train_cleaned = opschonen_dataframe(df=df_train, config=config)
    df_test_cleaned = opschonen_dataframe(df=df_test, config=config)
    
    return df_train_cleaned, df_test_cleaned


def numeriek_maken_data(df_train_clean, df_test_clean, config=_config):
    
    df_train_num = numeriek_maken_dataframe(df=df_train_clean, config=config)
    df_test_num = numeriek_maken_dataframe(df=df_test_clean, config=config)
    
    return df_train_num, df_test_num

def voeg_passagiers_toe(df_train, df_test, config=_config):

    #steps:
    # 1. Add column for 'workshop_passagiers' in both dataframes (all value 0)
    # 2. Collect new records
    # 3. Add records to both dataframes (with column 'workshop_passagiers'=1)
    
    df_train_added = df_train.copy()
    df_test_added = df_test.copy()

    # df_train_added['Workshop_passagier'] = 0
    # df_test_added['Workshop_passagier'] = 0

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