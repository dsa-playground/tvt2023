# Imports
import numpy as np
# from sklearn.preprocessing import LabelEncoder


def filter_isin(item_to_filter, compare_list):
    """Filter list from specific item.

    Parameters
    ----------
    item_to_filter : str/int
        String or int of a value that shouldn't be in list
    compare_list : list
        List with items to check

    Returns
    -------
    list
        List with all items except the specific one.
    """
    if isinstance(item_to_filter, list):
        return [key for key in item_to_filter if key in compare_list]
    if isinstance(item_to_filter, dict):
        return {k: v for k, v in item_to_filter.items() if k in compare_list}
    

def rename_cols(df, dict_renamed_columns):
    """Rename all columns if available.

    Note: In this function only matching columns with the keys will be renamed and returned!

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with number of columns.
    dict_renamed_columns : dict
        Dictionary with as key the current colname and as value the new columnname.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
    """
    dict_renamed_columns = filter_isin(
        item_to_filter=dict_renamed_columns,
        compare_list=df.columns
    )
    return df.rename(columns=dict_renamed_columns)


def change_order_cols(df, list_order_cols):
    """Change order of columns.
    
    Note: In this function only avaliable columns will be ordered and returned!

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with number of columns.
    list_order_cols : list(str)
        List of strings with columnnames.

    Returns
    -------
    pd.DataFrame
        DataFrame with the ordered columns.
    """
    list_order_cols = filter_isin(
        item_to_filter=list_order_cols,
        compare_list=df.columns
    )
    return df[list_order_cols]


def add_sum_column(df, variable_name, list_columns_to_sum):
    """Add a summerized column from a number of columns.

    Note: In this function only avaliable columns will be summerized in new column!

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a number of numeric columns.
    variable_name : str
        Name of the summerized column.
    list_columns_to_sum : list(str)
        List with names of columns to summerize

    Returns
    -------
    pd.DataFrame
        DataFrame with a new sum column.
    """
    list_columns_to_sum = filter_isin(
        item_to_filter=list_columns_to_sum,
        compare_list=df.columns
    )
    df[variable_name]=df[list_columns_to_sum].sum(axis=1)
    return df


def drop_cols(df, list_drop_cols):
    """Drop columns from DataFrame.

    Note: In this function only avaliable columns will be dropped from DataFrame!

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with number of columns.
    list_drop_cols : list(str)
        List with columnnames to be dropped.

    Returns
    -------
    pd.DataFrame
        DataFrame without number of columns.
    """
    list_drop_cols = filter_isin(
        item_to_filter=list_drop_cols,
        compare_list=df.columns
    )
    return df.drop(columns=list_drop_cols)

    
def replace_values_cols(df, dict_replacing):
    """Replace values of columns based of dictionary.

    Note: In this function only the values of matching columns with the keys will be replaced!

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with number of columns
    dict_replacing : dict(str,dict)
        Dictionary with columnname as key and dictionary of old/new value. 

    Returns
    -------
    pd.DataFrame
        DataFrame with replaced values.
    """
    dict_replacing = filter_isin(
        item_to_filter=dict_replacing,
        compare_list=df.columns
    )
    return df.replace(dict_replacing)


def fillna_cols(df, dict_fillna_strategy):
    """Fills NaN values in columns with certain strategy.

    Note: In this function only the values of matching columns with the keys will be filled!

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a number of columns.
    dict_fillna_strategy : dict
        Dictionary with as columnname as key and strategy as value. 
        Possible strategies: mean, backfill, bfill, pad, ffill

    Returns
    -------
    pd.DataFrame
        DataFrame with filled missing values for certain columns.
    """
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
    """Drops rows with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with number of columns.

    Returns
    -------
    pd.DataFrame
        DataFrame without rows where one or multiple columns have a missing value.
    """
    return df.dropna(axis=0)


def floor_values_cols(df, list_rounding_cols):
    """Floor values of a given list of columnnames.

    Note: In this function only avaliable columns will be floored from DataFrame!

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with number of columns.
    list_rounding_cols : list(str)
        List of strings with the columnnames that need to be floored.

    Returns
    -------
    pd.DataFrame
        DataFrame with floored columns.
    """
    list_rounding_cols = filter_isin(
        item_to_filter=list_rounding_cols,
        compare_list=df.columns
    )
    df[list_rounding_cols] = df[list_rounding_cols].apply(np.floor).astype('Int64')
    return df


def create_index(df, list_index_cols):
    """Creates index from column(s).

    Note: In this function only avaliable columns will be used in index!

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with number of columns.
    list_index_cols : list(str)
        List of strings with the columnnames that need to be used in the index.

    Returns
    -------
    pd.DataFrame
        DataFrame with new index.
    """
    list_index_cols = filter_isin(
        item_to_filter=list_index_cols,
        compare_list=df.columns
    )
    return df.set_index(list_index_cols)


# # Old function, previous used to change column values, later changed by using function replace_values_cols
# # for clarity/readability. This function requires import:
# # from sklearn.preprocessing import LabelEncoder
# def label_encode_cols(df, list_label_encode_cols):
#     """Encode column values with different values.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame with number of columns.
#     list_label_encode_cols : list(str)
#         List with number of columns to encode

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with encoded columns.
#     """
#     list_label_encode_cols = filter_isin(
#         item_to_filter=list_label_encode_cols,
#         compare_list=df.columns
#     )
#     l1=LabelEncoder()
#     for col in list_label_encode_cols:
#         df[col]=l1.fit_transform(df[col])
#     return df
