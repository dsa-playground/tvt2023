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
    """_summary_

    Parameters
    ----------
    config : _type_, optional
        _description_, by default _config

    Returns
    -------
    _type_
        _description_
    """
    x = input('Enter your name:')
    return x


def collect_int_input(question, max=10, min=0):
    """Collect integer input given a question and range.

    Parameters
    ----------
    question : str
        Question that will be printed before the 'input()' command.
    max : int, optional
        Integer value that determines the maximum (included), by default 10
    min : int, optional
        Integer value that determines the minimum (included), by default 0

    Returns
    -------
    int
        Integer value collected from the user.
    """
    print(question)
    found = False

    while not found:
        try: 
            myInt = int(input("Antwoord: "), 10)
        except Exception:
            print('Geef een geheel getal.')
            continue

        if min <= myInt <= max:
            return myInt
        elif myInt > max:
            print(f"Is een cijfer van hoger dan {max} realistisch?")
        elif myInt < min:
            print(f"Is een cijfer van lager dan {min} realistisch?")


def collect_str_input(question, possible_entries=[]):
    """Collect string input given a question and possible entries (restrictions).

    Parameters
    ----------
    question : str
        Question that will be printed before the 'input()' command.
    possible_entries : list, optional
        List of strings that are allowed, by default all entries are allowed.

    Returns
    -------
    str
        String value (lowercase) collected from the user.

    Raises
    ------
    ValueError
        ValueError will be raised if string value is empty.
    """
    print(question)
    possible_entries = [entry.lower() for entry in possible_entries]
    found = False

    while not found:
        try: 
            myStr = input("Antwoord: ").lower()
            if not (myStr and myStr.strip()):
                raise ValueError('Leeg veld.')
        except Exception:
            print('Een leeg antwoord is niet bruikbaar.')
            continue
        
        if len(possible_entries)>0:
            if myStr in possible_entries:
                return myStr
            else:
                print(f"Je antwoord moet een van de volgende opties zijn: {possible_entries}.")
        else:
          return myStr

def test_str_input():
    name = give_str_input(
        question="Wat is je naam?")
    sex = give_str_input(
        question="Wat is je geslacht (man, vrouw, neutraal)?",
        possible_entries=['man', 'vrouw', 'neutraal'])
    return [name, sex]


def add_records(config=_config):
    name = collect_str_input(
        question='Wat is je naam?')
    age = collect_int_input(
        question='Vul hier je leeftijd in:',
        max=100,
        min=0)
    sex = collect_str_input(
        question='Wat is je geslacht (man, vrouw, neutraal)?',
        possible_entries=['man', 'vrouw', 'neutraal'])
    kids = collect_int_input(
        question='Hoeveel kinderen neem je mee op reis?',
        max=10,
        min=0)
    family = collect_int_input(
        question='Hoeveel familieleden gaan mee op reis gaan?',
        max=10,
        min=0)
    multi = collect_str_input(
        question="""
            Geef aan welke optie je voorkeur geniet voor de overige variabelen:
            A. Frankrijk, 1e klasse
            B. Engeland, 1e klasse
            C. Ierland, 1e klasse
            """,
        possible_entries=['a', 'b', 'c'])
    multi = collect_str_input(
        question="Wil je nog een passagier toevoegen?",
        possible_entries=['ja', 'nee', 'j', 'n'])
    return [name, age, sex, kids, family, multi]


def spielerij_entry():
    a = input("Flauwekul antwoord:")
    b = input("Doorgaan met meer vragen?")
    return [a,b]


def voeg_passagiers_toe(all_records):

    new_record = spielerij_entry()
    print(new_record)
    if new_record[-1] in ['ja', 'j']:
        all_records=all_records + new_record[:-1]
        print(all_records)
        voeg_passagiers_toe(all_records=all_records)
    elif new_record[-1] in ['nee', 'n']:
        all_records=all_records + new_record[:-1]
        print(all_records)
        return all_records

# def voeg_passagiers_toe(all_records=[]):

#     new_record = spielerij_entry()
#     # print(new_record)
#     if new_record[-1] in ['ja', 'j']:
#         all_records.append(new_record[:-1])
#         # print(all_records)
#         voeg_passagiers_toe(all_records=all_records)
#     elif new_record[-1] in ['nee', 'n']:
#         all_records.append(new_record[:-1])
#         # print("WTF")
#         print(type(all_records))
#         return all_records