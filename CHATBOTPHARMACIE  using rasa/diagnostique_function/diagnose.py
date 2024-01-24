import logging
import pandas as pd
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('fr_core_news_md')
diagnosis_df = pd.read_pickle("data/diagnosis_data.pkl")
symptoms_df = pd.read_pickle("data/symptoms.pkl")

# logging config
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename='logging.log',
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG
)


def encode_symptom(symptom):
    '''
    Convert symptom string to vector using spacy

    :param symptom:
    :return: 256-D vector
    '''

    logging.info(f"Encoding symptom {symptom}")
    encoded_symptom = nlp(symptom).vector.tolist()

    return encoded_symptom


def create_illness_vector(encoded_symptoms):
    '''
    Compares the list of encoded symptoms to a list of encoded symptoms. Any symptom above threshold (0.85) will be
    flagged.

    :param encoded_symptoms: A list of encoded symptoms
    :return: A single vector flagging each symptoms appearence in the user message (based on vector similarity)
    '''

    threshold = 0.60
    symptoms_df['symptom_flagged'] = 0

    for encoded_symptom in encoded_symptoms:

        symptoms_df['similarity'] = list(cosine_similarity(np.array(encoded_symptom).reshape(1, -1),
                                                           np.array(list(symptoms_df['symptom_vector'])))[0])

        symptoms_df.loc[symptoms_df['similarity'] > threshold, 'symptom_flagged'] = 1

        number_of_symptoms_flagged = len(symptoms_df.loc[symptoms_df['similarity'] > threshold, 'symptom_flagged'])

        logging.info(f"Flagged {number_of_symptoms_flagged} potential symptom matches")

    return list(symptoms_df['symptom_flagged'])


def get_diagnosis(illness_vector):
    '''
    Compares the symptoms vector to our diagnosis df and generate the diagnosis (if one exists)

    :param illness_vector:
    :return: A string containing the diagnosis based off of illness vector similarity
    '''

    threshold = 0.40

    diagnosis_df['similarity'] = list(cosine_similarity(np.array(illness_vector).reshape(1, -1),
                                                        np.array(list(diagnosis_df['illness_vector'])))[0])

    # If there is an illness (or multiple illnesses)
    if len(diagnosis_df.loc[diagnosis_df['similarity'] > threshold]) > 0:
        illness = (
            diagnosis_df
            .sort_values(by='similarity', ascending=False)['illness']
            .iloc[0]
        )

        logging.info(f"Diagnostic utilisateur avec {illness}")
        diagnosis_string = f"D'après vos symptômes, il semble que vous pourriez avoir {illness}"

    else:
        closest_match = (
            diagnosis_df
            .sort_values(by='similarity', ascending=False)[['illness', 'similarity']]
            .head(1)
        )
        logging.info(f"Impossible de trouver un diagnostic, la correspondance la plus proche était {closest_match['illness'].iloc[0]} "
                     f"at {closest_match['similarity'].iloc[0]}")
        diagnosis_string = "Malheureusement, je ne suis pas en mesure de vous diagnostiquer sur la base des symptômes que vous avez fournis"

    return diagnosis_string
