# Imports
import numpy as np
# from sklearn.preprocessing import LabelEncoder


def filter_isin(item_to_filter, compare_list):
    if isinstance(item_to_filter, list):
        return [key for key in item_to_filter if key in compare_list]
    if isinstance(item_to_filter, dict):
        return {k: v for k, v in item_to_filter.items() if k in compare_list}
    

def rename_cols(df, dict_renamed_columns):
    dict_renamed_columns = filter_isin(
        item_to_filter=dict_renamed_columns,
        compare_list=df.columns
    )
    return df.rename(columns=dict_renamed_columns)


def change_order_cols(df, list_order_cols):
    list_order_cols = filter_isin(
        item_to_filter=list_order_cols,
        compare_list=df.columns
    )
    return df[list_order_cols]


def drop_cols(df, list_drop_cols):
    list_drop_cols = filter_isin(
        item_to_filter=list_drop_cols,
        compare_list=df.columns
    )
    return df.drop(columns=list_drop_cols)

    
def replace_values_cols(df, dict_replacing):
    dict_replacing = filter_isin(
        item_to_filter=dict_replacing,
        compare_list=df.columns
    )
    return df.replace(dict_replacing)


def fillna_cols(df, dict_fillna_strategy):
    dict_fillna_strategy = filter_isin(
        item_to_filter=dict_fillna_strategy,
        compare_list=df.columns
    )
    for col in dict_fillna_strategy.keys():
        if dict_fillna_strategy[col] == 'mean':
            df[col]=df[col].fillna(value=df[col].mean())
        if dict_fillna_strategy[col] in ['backfill', 'bfill', 'pad', 'ffill']:
            df[col]=df[col].fillna(method=dict_fillna_strategy[col])
    return df


def dropna_rows(df):
    return df.dropna(axis=0)


def floor_values_cols(df, list_rounding_cols):
    list_rounding_cols = filter_isin(
        item_to_filter=list_rounding_cols,
        compare_list=df.columns
    )
    df[list_rounding_cols] = df[list_rounding_cols].apply(np.floor).astype('Int64')
    return df


# def label_encode_cols(df, list_label_encode_cols):
#     list_label_encode_cols = filter_isin(
#         item_to_filter=list_label_encode_cols,
#         compare_list=df.columns
#     )
#     l1=LabelEncoder()
#     for col in list_label_encode_cols:
#         df[col]=l1.fit_transform(df[col])
#     return df


def create_index(df, list_index_cols):
    list_index_cols = filter_isin(
        item_to_filter=list_index_cols,
        compare_list=df.columns
    )
    return df.set_index(list_index_cols)
