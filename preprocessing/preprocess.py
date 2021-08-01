"""
The preprocessing module. This module is used for cleaning, grouping, and formatting
the data so that it is readable by other modules further down the pipeline.
"""
import os
import re
import csv
import pickle
import pandas as pd
import numpy as np

DEMENTIABANK_INFO_PATH = '/home/thiago/BERTFineTuning/preprocessing/data/dementiabank_info'
DEMENTIABANK_PATH = '/home/thiago/BERTFineTuning/preprocessing/data/dementiabank'
PICKLE_PATH = '/home/thiago/BERTFineTuning/preprocessing/data/pickled_data/dbank.pkl'

def get_pickle_data():
    """
    Takes all valid transcript and patient info from the dementiabank dataset,
    and puts it into 
    
    {'pid':[], 'text':[], 'label':[], 'age':[], 'gender':[]}

    formatting, where ``pid`` is the patient ID, ``text`` is the contents of the transcript, 
    ``label`` is 0 if the patient is a control else 1, ``age`` is the age of the patient, and 
    ``gender`` is the gender of the patient.

    The dictionary is saved as dbank.pkl under ``PICKLE_PATH``.

    Returns
    -------
    dict
    """
    parsed_data = {"pid":[], "text":[], "label":[], "age":[], "gender":[]}
    diagnoses = get_diagnoses(os.path.join(DEMENTIABANK_INFO_PATH, 'diagnosis.txt'))
    ages, genders = get_age_gender(os.path.join(DEMENTIABANK_INFO_PATH, 'age_gender.txt'))

    for filename in os.listdir(DEMENTIABANK_PATH):
        if filename.endswith('.txt'):
            try:
                pid = int(filename[0:3])
                diag = diagnoses[filename.rstrip('.txt')]
                age = ages[filename.rstrip('.txt')]
                gender = genders[filename.rstrip('.txt')]
        
                if diag == 'Control':
                    label = 0
                else:
                    label = 1

                with open(os.path.join(DEMENTIABANK_PATH, filename)) as file:
                    text = ''
                    for line in file:
                        text = text + line
                
                text = clean_text(text)
                parsed_data["pid"].append(pid)
                parsed_data["text"].append(text)
                parsed_data["label"].append(label)
                parsed_data["age"].append(age)
                parsed_data["gender"].append(gender)
            except:
                print('Error occured with subject {}'.format(filename.rstrip('.txt')))

    with open('/home/thiago/BERTFineTuning/preprocessing/data/pickled_data/dbank.pkl', 'wb') as outfile:
        pickle.dump(parsed_data, outfile, protocol=2)

    return parsed_data

def get_diagnoses(filename):
    """
    Returns
    -------
    dict
        a diagnosis for each subject
    """
    diag = {}
    with open(filename) as file:
        for line in file:
            l = line.split()
            diag[l[0].rstrip("c")] = l[1]
    diag.pop('interview')
    return diag

def get_age_gender(filename):
    """
    Returns
    -------
    dict 
        an age for each subject

    dict
        a gender for each subject
    """
    ages = {}
    genders = {}
    with open(filename) as file:
        for line in file:
            l = line.split()
            ages[l[0].rstrip("c")] = l[1]
            genders[l[0].rstrip("c")] = l[2]
    ages.pop('interview')
    genders.pop('interview')
    return ages, genders

def clean_text(text):
    """
    Cleans texts by removing non ascii characters.

    Returns
    -------
    str
    """
    text = text.strip()
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = text.replace("\n", "")
    return text