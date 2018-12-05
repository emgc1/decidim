import constants
from google.cloud import translate
from pathlib import Path
import pandas as pd
import traceback
from time import sleep

CHUNK_SIZE = 1000
# Columns to translate in proposal file
COLUMNS = ['Title ES', 'Title CA', 'Description ES', 'Description CA']

#Google translate parameter
FORMAT = 'text'

# Mapping between original column name and result column name per language to translate
# OUTPUTS = {'Title': ['Title ES', 'Title CA'], 'Description': ['Description ES', 'Description CA']}
OUTPUTS = {'Category': ['Category EN'], 'Subcategory': ['Subcategory EN']}
# OUTPUTS = {'Title ES': ['Title_ES_EN', 'Title_ES_CAT'], 'Title CA': ['Title_CA_EN', 'Title_CA_ES'],
#            'Description ES': ['Description_ES_EN', 'Description_ES_CAT'], 'Description CA': ['Description_CA_EN', 'Description_CA_ES']}

# Original and target language, it will iterate over every pair in the list, for this case in the
# first iteration will send the request to googlecloud with origin language catalan and target language spanish
# for the second iteration origin language spanish and target language catalan
# LANGUAGES_MAP = {
#     'Title ES': [['es', 'en'], ['es', 'ca']],
#     'Title CA': [['ca', 'en'], ['ca', 'es']],
#     'Description ES': [['es', 'en'], ['es', 'ca']],
#     'Description CA': [['ca', 'en'], ['ca', 'es']]
# }

LANGUAGES_MAP = {
    'Category': [[None, 'en']],
    'Subcategory': [[None, 'en']]
}

translated = {}
translate_client = translate.Client()

def load_data():
    if not Path(constants.PROPOSAL_EN_FULL_RESULT_PATH).is_file():
        df = pd.read_csv(constants.PROPOSALS_EN_PATH, encoding='utf-8')
        for _, value in OUTPUTS.iteritems():
            df[value[0]] = None
            # df[value[1]] = None
        df.to_csv(open(constants.PROPOSAL_EN_FULL_RESULT_PATH, 'w'), encoding='utf-8')
    else:
        df = pd.read_csv(constants.PROPOSAL_EN_FULL_RESULT_PATH, encoding='utf-8')
    return df


def save_data(df):
    full_result_out = open(constants.PROPOSAL_EN_FULL_RESULT_PATH, 'w')
    df.to_csv(full_result_out, encoding='utf-8')


def partial_translation(df):
    df_rows = df.shape[0]
    start_row = df[df['Category EN'].isnull() | df['Subcategory EN'].isnull() | df['Scope EN'].isnull()].index[0]
    try:
        while start_row < df_rows:
            sub_df = df.iloc[start_row:start_row + CHUNK_SIZE]
            start_row += CHUNK_SIZE
            for idx in range(0, 1):
                for col in OUTPUTS.iterkeys():  # To Translate Title in the first iteration and Description in the second
                    to_translate = []
                    chr_count = 0
                    for index, row in sub_df.iterrows():
                        if pd.isnull(row[OUTPUTS[col][idx]]):
                            chr_count += len(row[col])
                            if row[col] not in translated:
                                if chr_count < 100000:
                                    to_translate.append(row[col])  # Appending the text to translate from the select column
                                else:
                                    translation = call_google_translate(to_translate, LANGUAGES_MAP[col][idx][1], LANGUAGES_MAP[col][idx][0])
                                    add_result(df, col, OUTPUTS[col][idx], translation)
                                    to_translate = [row[col]]
                                    chr_count = 0
                            else:
                                add_single_result(df, col, OUTPUTS[col][idx], row[col], translated[row[col]])

                    if to_translate:
                        translation = call_google_translate(to_translate, LANGUAGES_MAP[col][idx][1], LANGUAGES_MAP[col][idx][0])
                        add_result(df, col, OUTPUTS[col][idx], translation)

            save_data(df)
    except Exception as e:
        traceback.print_exc()
        print "Unexpected error: {}".format(e.message)
        save_data(df)


def call_google_translate(to_translate, target, source=None):
    translation = None
    to_translate = list(set(to_translate))
    while translation is None:
        try:
            if source:
                translation = translate(to_translate, source, target)
            else:
                translation = detect_translate(to_translate, target)
        except Exception as e:
            print e.message
            sleep(5)  # To avoid exceed user limit rate

    return translation


def add_result(df, input_col, output_col, translation):
    for result in translation:
        translated[result['input']] = result['translatedText']
        df.loc[df.loc[:, input_col] == result['input'], [output_col]] = result['translatedText']


def add_single_result(df, input_col, output_col, input_text, output_text):
    df.loc[df.loc[:, input_col] == input_text, [output_col]] = output_text


def translate(to_translate, source, target):
    return translate_client.translate(
        to_translate,
        source_language=source,
        target_language=target,
        format_=FORMAT)


def detect_translate(to_translate, target):
    return translate_client.translate(
        to_translate,
        target_language=target,
        format_=FORMAT)

if __name__ == '__main__':
    partial_translation(load_data())
