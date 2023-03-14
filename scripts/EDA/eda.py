# Imports
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio

from pyaml_env import parse_config

# Load settings
config_path="../settings.yml"

_config = parse_config(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))

pio.templates.default = _config["EDA"]["visualisation"]["plotly_template_default"]
color_discrete_map= _config["EDA"]["visualisation"]["color_discrete_map"]
category_orders = _config["EDA"]["visualisation"]["category_orders"]

# Functions

def create_df_count(df,columns):
    df_count = df.groupby(columns).size().reset_index()
    df_count.columns = columns + ["Aantal"]
    return df_count

def create_df_percentage(df,groupby_columns,percentage_column):
    percentage = df.groupby(groupby_columns)[percentage_column].value_counts(normalize = True)
    df_percentage = pd.DataFrame(percentage)
    df_percentage = df_percentage * 100
    df_percentage = df_percentage.rename({percentage_column: "Percentage"}, axis='columns')
    df_percentage = df_percentage.reset_index()
    return df_percentage

def create_bar_plot(df,x,**kwargs):
    kwarg_optional_cols = ["color", "facet_col","facet_row"]
    
    kwarg_cols_result = set(kwarg_optional_cols) & set(list(kwargs.keys()))
    columns = [x]
    for col in kwarg_cols_result:
        columns.append(kwargs[col])

    if 'percentage' in list(kwargs.keys()):
        percentage_col = kwargs["percentage"]
        kwargs.pop('percentage', None)
        df = create_df_percentage(df=df,groupby_columns=columns,percentage_column=percentage_col)
        if 'color' in list(kwargs.keys()): 
            fig = px.bar(df, x=df[x], y=df["Percentage"],**kwargs)
        else:
            fig = px.bar(df, x=df[x], y=df["Percentage"],color=percentage_col,**kwargs)
    else:
        df = create_df_count(df=df,columns=columns)
        fig = px.bar(df, x=df[x], y=df["Aantal"],**kwargs)
    fig.show()

def create_scatter_plot(df,x,y,**kwargs):
    fig = px.scatter(df, x=df[x], y=df[y],**kwargs)
    fig.show() 

def create_histogram(df,x,**kwargs):
    fig = px.histogram(df, x=df[x],**kwargs).update_layout(yaxis_title="Aantal")
    fig.show() 

def EDA_visualisaties(df):
    create_scatter_plot(df=df,x="Passagier_Id",y="Leeftijd",color="Overleefd",color_discrete_map=color_discrete_map,category_orders=category_orders)
    create_bar_plot(df=df,x="Ticket_klasse",color="Overleefd",color_discrete_map=color_discrete_map,category_orders=category_orders)
    create_bar_plot(df=df,x="Opstapplaats",color="Overleefd", facet_col="Geslacht",color_discrete_map=color_discrete_map,category_orders=category_orders)
    create_bar_plot(df=df,x="Aantal_overige_familieleden", percentage="Overleefd",color_discrete_map=color_discrete_map,category_orders=category_orders)

def basis_feiten(df):
    passengers = df["Overleefd"]
    passengers_survived = passengers.loc[df["Overleefd"] == "Ja"]
    rate_survivors = 100 * len(passengers_survived)/len(passengers)

    average_age = df["Leeftijd"].mean()
    mode_embarked = df["Opstapplaats"].mode()[0]
    # women = df_train_clean.loc[df_train_clean["Geslacht"] == 'Vrouw']
    # women_survived = women.loc[df_train_clean["Overleefd"] == "Ja"]
    # rate_women = 100 * len(women_survived)/len(women)

    # men = df_train_clean.loc[df_train_clean["Geslacht"] == 'Man']
    # men_survived = men.loc[df_train_clean["Overleefd"] == "Ja"]
    # rate_men = 100 * len(men_survived)/len(men)

    print(f"Er zitten {len(passengers)} passagiers in de dataset, daarvan heeft {rate_survivors:.2f}% het overleefd.")
    print(f"De gemiddelde leeftijd van de passagiers is {average_age:.0f}.")
    print(f"De meeste passagiers zijn opgestapt in {mode_embarked}.")
    # print(f"Er zitten {len(men)} mannen in de dataset, daarvan heeft {rate_men:.2f}% het overleefd.")
    # print(f"Er zitten {len(women)} vrouwen in de dataset, daarvan heeft {rate_women:.2f}% het overleefd.")