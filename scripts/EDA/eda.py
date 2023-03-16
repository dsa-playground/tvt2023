# Imports
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff

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

def create_3d_scatter_plot(df,x,y,z,**kwargs):
    fig = px.scatter_3d(df, x=x, y=y, z=z,**kwargs)
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        )
    )
    fig.update_traces(marker_size = 3) # changed to see multiple layers

    # Settings for (start) camera settings
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=2.5)
    )

    fig.update_layout(scene_camera=camera)

    fig.show()

def create_histogram(df,x,**kwargs):
    fig = px.histogram(df, x=df[x],**kwargs).update_layout(yaxis_title="Aantal")
    fig.show() 

def EDA_visualisaties(df):
    create_bar_plot(df=df,x="Ticket_klasse",color="Overleefd",color_discrete_map=color_discrete_map,category_orders=category_orders)
    create_bar_plot(df=df,x="Opstapplaats",color="Overleefd", facet_col="Geslacht",color_discrete_map=color_discrete_map,category_orders=category_orders)
    create_scatter_plot(df=df,x="Passagier_Id",y="Leeftijd",color="Overleefd",color_discrete_map=color_discrete_map,category_orders=category_orders)
    create_3d_scatter_plot(df=df,x="Passagier_Id", y="Leeftijd", z="Geslacht",color="Overleefd",color_discrete_map=color_discrete_map, 
                           category_orders=category_orders,hover_data={"Passagier_Id": False})
    create_bar_plot(df=df,x="Aantal_familieleden", percentage="Overleefd",color_discrete_map=color_discrete_map,category_orders=category_orders)

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

def correlatie_heatmap(df):
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    df_mask = corr.mask(mask)
    # df_mask = df_mask.dropna(axis=1, how='all')
    # df_mask = df_mask.dropna(axis=0, how='all')
    df_mask = df_mask.round(2)

    fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
                                    x=df_mask.columns.tolist(),
                                    y=df_mask.columns.tolist(),
                                    colorscale=px.colors.diverging.RdBu,
                                    font_colors=['black'],
                                    hoverinfo="none", #Shows hoverinfo for null values
                                    showscale=True, ygap=1, xgap=1
                                    )

    fig.update_xaxes(side="bottom")

    fig.update_layout(
        title_text='Correlatie heatmap', 
        title_x=0.5, 
        width=1000, 
        height=1000,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed',
        template='plotly_white',
    )

    # NaN values are not handled automatically and are displayed in the figure
    # So we need to get rid of the text manually
    for i in range(len(fig.layout.annotations)):
        if fig.layout.annotations[i].text == 'nan':
            fig.layout.annotations[i].text = ""

    fig.show()
