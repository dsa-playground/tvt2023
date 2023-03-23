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
width = _config["EDA"]["visualisation"]["plot_width"]

# Functions

def create_df_count(df,columns):
    """Create DataFrame with counts of columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to count
    columns : list(str)
        List of strings with columnnames.

    Returns
    -------
    pd.DataFrame
        DataFrame with counts of records for group of columns.
    """
    df_count = df.groupby(columns).size().reset_index()
    df_count.columns = columns + ["Aantal"]
    return df_count


def create_df_percentage(df,groupby_columns,percentage_column):
    """Create DataFrame with percentages of columns based on specific 'percentage column'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns listed in groupby_columns and percentage_column
    groupby_columns : list(str)
        List of columns to groupby on.
    percentage_column : str
        Columnname to calculate the percentage.

    Returns
    -------
    pd.DataFrame
        DataFrame with the percentage of a column given the input DataFrame.
    """
    percentage = df.groupby(groupby_columns)[percentage_column].value_counts(normalize = True)
    df_percentage = pd.DataFrame(percentage)
    df_percentage = df_percentage * 100
    df_percentage = df_percentage.rename({percentage_column: "Percentage"}, axis='columns')
    df_percentage = df_percentage.reset_index()
    return df_percentage


def create_bar_plot(df,x,**kwargs):
    """Creates Plotly bar plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to incorporate in the bar plot
    x : str
        Columnname to use on the X-axis

    Returns
    -------
    Plotly visualisation
    """
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
    """Create Plotly scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to incorporate in the scatter plot
    x : str
        Columnname of the X-axis
    y : str
        Columnname of the Y-axis
    
    Returns
    -------
    Plotly visualisation
    """
    fig = px.scatter(df, x=df[x], y=df[y],**kwargs)
    fig.show() 


def create_3d_scatter_plot(df,x,y,z,**kwargs):
    """Create Plotly 3D scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to incorporate in the 3D scatter plot
    x : str
        Columnname of the X-axis
    y : str
        Columnname of the Y-axis
    z : str
        Columnname of the Z-axis

    Returns
    -------
    Plotly visualisation
    """
    fig = px.scatter_3d(df, x=x, y=y, z=z,**kwargs)
    # Settings for (start) camera settings
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=2.5)
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        ),
        scene_camera=camera
    )
    fig.update_traces(marker_size = 4) # changed to see multiple layers
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    fig.show()


def create_histogram(df,x,**kwargs):
    """Create Plotly histogram.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to incorporate in the histogram.
    x : str
        Columnname of the X-axis
        
    Returns
    -------
    Plotly visualisation
    """
    fig = px.histogram(df, x=df[x],**kwargs).update_layout(yaxis_title="Aantal")
    fig.show() 

def visualisatie_ticketklasse(df):
    """Create Barplot of number of passengers survived/passed away (stacked) given the ticket class.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to incorporate in the histogram.
    
    Returns
    -------
    Plotly visualisation
    """
    print("Het aantal overleefde/overleden passagiers ten opzichte van ticketklasse.")
    create_bar_plot(df=df,x="Ticket_klasse",color="Overleefd",color_discrete_map=color_discrete_map,category_orders=category_orders,width=width,
                    )

def visualisatie_opstapplaats(df):
    """Create Barplot of the number of passengers survived/passed away (stacked) given the boarding place and gender.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to incorporate in the histogram.
    
    Returns
    -------
    Plotly visualisation
    """
    print("Het aantal overleefde/overleden passagiers ten opzichte van opstapplaats en geslacht.")
    create_bar_plot(df=df,x="Opstapplaats",color="Overleefd", facet_col="Geslacht",color_discrete_map=color_discrete_map,category_orders=category_orders,width=width)
                    
def visualisatie_leeftijd(df):
    """Create Scatterplot of survived/passed away passengers given age.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to incorporate in the histogram.
    
    Returns
    -------
    Plotly visualisation
    """   
    print("Scatterplot van de overleefde/overleden passagiers ten opzichte van leeftijd.")
    create_scatter_plot(df=df,x="Passagier_Id",y="Leeftijd",color="Overleefd",color_discrete_map=color_discrete_map,category_orders=category_orders,width=width)

def visualisatie_leeftijd_geslacht(df):
    """Create 3D Scatterplot of survived/passed away passengers given age and gender.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to incorporate in the histogram.
    
    Returns
    -------
    Plotly visualisation
    """    
    print("3D scatterplot van de overleefde/overleden passagiers ten opzichte van leeftijd en geslacht.")
    create_3d_scatter_plot(df=df,x="Passagier_Id", y="Leeftijd", z="Geslacht",color="Overleefd",color_discrete_map=color_discrete_map, 
                           category_orders=category_orders,hover_data={"Passagier_Id": False},width=width)

def visualisatie_familieleden(df):
    """Create Barplot of percentage of passengers survived/passed away (stacked) given the number of family members.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to incorporate in the histogram.
    
    Returns
    -------
    Plotly visualisation
    """
    print("Het percentage overleefde/overleden passagiers ten opzichte van het aantal familieleden.")
    create_bar_plot(df=df,x="Totaal_aantal_familieleden", percentage="Overleefd",color_discrete_map=color_discrete_map,category_orders=category_orders,width=width)

def basis_feiten(df):
    """Prints several interesting facts about the titanic dataset, including missing values.

    Facts returned:
    1. Number of passengers, including percentage that survived.  
    2. Average age of passengers.
    3. Place where most passengers embarked.
    4. Number of missing values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to incorporate in the print statements.

    Returns
    -------
    Printed facts
    """
    passengers = df["Overleefd"]
    passengers_survived = passengers.loc[df["Overleefd"] == "Ja"]
    rate_survivors = 100 * len(passengers_survived)/len(passengers)

    average_age = df["Leeftijd"].mean()
    mode_embarked = df["Opstapplaats"].mode()[0]

    # Determine missing values in %
    percent_missing = df.isnull().sum() * 100 / len(df)
    df_missing_value = pd.DataFrame({'Missende waarden (%)': percent_missing})

    print(f"Er zitten {len(passengers)} passagiers in de dataset, daarvan heeft {rate_survivors:.2f}% het overleefd.")
    print(f"De gemiddelde leeftijd van de passagiers is {average_age:.0f}.")
    print(f"De meeste passagiers zijn opgestapt in {mode_embarked}.")
    print(f" ")
    display(df_missing_value.sort_values(by='Missende waarden (%)', ascending=False))
    print(f" ")

def correlatie_heatmap(df):
    """Shows correlation heatmap from a dataframe.

    Note: only numeric columns are incorporated in the heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to incorporate in the correlation heatmap.

    Returns
    -------
    Correlation heatmap visualisation.
    """
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    df_mask = corr.mask(mask)
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
