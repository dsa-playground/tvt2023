import os
import pandas as pd
from pyaml_env import parse_config
config_path="../settings.yml"

_config = parse_config(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))


def collect_int_input(question, boundaries=[]):
    """Collect integer input given a question and range.

    Parameters
    ----------
    question : str
        Question that will be printed before the 'input()' command.
    boundaries : list, optional
        List with two integer values that determine the range within integer should be:
            * minimum (boundaries[0])
            * maximum (boundaries[1])

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
        
        if len(boundaries)>0:
            if boundaries[0] <= myInt <= boundaries[1]:
                return myInt
            elif myInt > boundaries[1]:
                print(f"Is een cijfer van hoger dan {max} realistisch?")
            elif myInt < boundaries[0]:
                print(f"Is een cijfer van lager dan {min} realistisch?")
        else:
            return myInt


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

    possible_entries = [entry.lower() for entry in possible_entries if isinstance(entry, str)]
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


def add_record(config=_config):
    answers = {}
    for item in config['preprocess']['data']['collect']['items_to_collect']:
        if config['preprocess']['data']['collect']['items_to_collect'][item]['type'] == 'str':
            answer = collect_str_input(
                question=config['preprocess']['data']['collect']['items_to_collect'][item]['question'],
                possible_entries=config['preprocess']['data']['collect']['items_to_collect'][item]['restriction'])
        elif config['preprocess']['data']['collect']['items_to_collect'][item]['type'] == 'int':
            answer = collect_int_input(
                question=config['preprocess']['data']['collect']['items_to_collect'][item]['question'],
                boundaries=config['preprocess']['data']['collect']['items_to_collect'][item]['restriction'])
        answers[item] = answer
    
    return answers

def add_multiple_records(continue_key='add', all_records=[]):

    new_record = add_record()
    # print(f"start: {all_records}")
    if new_record[continue_key] in ['ja', 'j']:
        all_records.append(new_record)
        # print(f"if loop: {all_records}")
        return add_multiple_records(all_records=all_records)
    else:
        # print(f"elif loop: {all_records}")
        all_records.append(new_record)
        # print(f"elif loop + : {all_records}")
        return all_records


def transform_multiplechoice_anwser(list_with_dicts, config=_config):
    updated_list = []
    for item in list_with_dicts:
        updated_dict = {**item, **config['preprocess']['data']['collect']['transform_multi'][item['multi']]}
        updated_list.append(updated_dict)
    return updated_list


def transform_multi_records_to_df(list_with_all_new_records):
    df_new_records = pd.DataFrame(list_with_all_new_records)
    df_new_records['Passagier_Id'] = df_new_records.index+10_000
    df_new_records['Workshop_passagier'] = 1
    df_new_records = df_new_records.drop(columns=['multi', 'add'])
    return df_new_records






# def add_records_ugly(config=_config):
#     name = collect_str_input(
#         question='Wat is je naam?')
#     age = collect_int_input(
#         question='Vul hier je leeftijd in:',
#         max=100,
#         min=0)
#     sex = collect_str_input(
#         question='Wat is je geslacht (man, vrouw, neutraal)?',
#         possible_entries=['man', 'vrouw', 'neutraal'])
#     kids = collect_int_input(
#         question='Hoeveel kinderen neem je mee op reis?',
#         max=10,
#         min=0)
#     family = collect_int_input(
#         question='Hoeveel familieleden gaan mee op reis gaan?',
#         max=10,
#         min=0)
#     multi = collect_str_input(
#         question="""
#             Geef aan welke optie je voorkeur geniet voor de overige variabelen:
#             A. Frankrijk, 1e klasse
#             B. Engeland, 1e klasse
#             C. Ierland, 1e klasse
#             """,
#         possible_entries=['a', 'b', 'c'])
#     multi = collect_str_input(
#         question="Wil je nog een passagier toevoegen?",
#         possible_entries=['ja', 'nee', 'j', 'n'])
#     return [name, age, sex, kids, family, multi]


# def spielerij_entry():
#     a = input("Flauwekul antwoord:")
#     b = input("Doorgaan met meer vragen?")
#     return [a,b]


# def test_str_input():
#     name = give_str_input(
#         question="Wat is je naam?")
#     sex = give_str_input(
#         question="Wat is je geslacht (man, vrouw, neutraal)?",
#         possible_entries=['man', 'vrouw', 'neutraal'])
#     return [name, sex]