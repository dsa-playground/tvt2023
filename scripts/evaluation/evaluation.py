# Imports
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
from pyaml_env import parse_config
from sklearn.ensemble import RandomForestClassifier

# Load settings
config_path="../settings.yml"

_config = parse_config(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))

pio.templates.default = _config["EDA"]["visualisation"]["plotly_template_default"]
width = _config["EDA"]["visualisation"]["plot_width"]

def calculate_feature_importance(df,config=_config):
    """Calculate the feature importance with a RandomForestClassifier.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with training data (X, y)
    config : dict, optional
        Dictionary with all settings, by default _config (which is loaded in this script).

    Returns
    -------
    pd.DataFrame
        DataFrame with in the index the feature names and a column 'Feature_importance' with values for the feature importance.
    """
    y_column=config['modeling']['y_variable']
    list_variables = list(set(df.columns) - set([y_column]))

    X=df[list_variables]
    y=df[y_column]

    clf = RandomForestClassifier(max_depth=2, random_state=0) 
    clf.fit(X, y)

    df_feature_importance = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=["Feature_importance"])
    df_feature_importance = df_feature_importance.sort_values(['Feature_importance'], ascending=False)

    return df_feature_importance

def geef_belangrijkste_variabelen(df):
    """Visualise the feature importance as a horizontal bar plot. 

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with training data (X, y)

    Returns
    -------
    Plotly visualisation: horizontal bar plot.
    """
    df_feature_importance = calculate_feature_importance(df)
    
    print("De variabelen welke de meeste invloed uitoefenen op het resultaat zijn:")
    
    fig = px.bar(df_feature_importance, x=df_feature_importance["Feature_importance"], y=df_feature_importance.index, width=width).update_yaxes(categoryorder="total ascending")
    fig.update_traces(marker_color='#00A9A4')
    fig.update_layout(margin=dict(l=300, r=100, b=100, t=100))
    fig.update_yaxes(visible=True, showticklabels=True,title="")
    fig.show()
