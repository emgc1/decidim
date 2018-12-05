import constants
from google.cloud import translate
import re
from pathlib import Path
import pandas as pd
import traceback
from time import sleep

CHUNK_SIZE = 100
FORMAT = 'text'
LANGUAGES = [['ca', 'es'], ['es', 'ca']]
translate_client = translate.Client()

# COLUMNS = ['Title', 'Description']
# COLUMNS_RESULT = ['Title ES', 'Title CA', 'Description ES', 'Description CA']
# OUTPUTS = {'Title': ['Title ES', 'Title CA'], 'Description': ['Description ES', 'Description CA']}
# FULL_PATH= constants.PROPOSAL_FULL_RESULT_PATH

COLUMNS = ['Title Result ES', 'Title Result CA', 'Description Result ES', 'Description Result CA']
OUTPUTS = {'Title Result ES': ['Title Result ES_EN', 'Title Result ES_CA'],
           'Title Result CA': ['Title Result CA_EN', 'Title Result CA_ES'],
           'Description Result ES': ['Description Result ES_EN', 'Description Result ES_CA'],
           'Description Result CA': ['Description Result CA_EN', 'Description Result CA_ES']}
FULL_PATH = constants.RESULTS_FULL_PATH

LANGUAGES_MAP = {
    'Title Result ES': [['es', 'en'], ['es', 'ca']],
    'Title Result CA': [['ca', 'en'], ['ca', 'es']],
    'Description Result ES': [['es', 'en'], ['es', 'ca']],
    'Description Result CA': [['ca', 'en'], ['ca', 'es']]
}


translate_client = translate.Client()

def load_data():
    if not Path(constants.RESULTS_FULL_PATH).is_file():
        df = pd.read_csv(constants.RESULTS_PATH, encoding='utf-8')
        for _, value in OUTPUTS.iteritems():
            df[value[0]] = None
            df[value[1]] = None
        df.to_csv(open(constants.RESULTS_FULL_PATH, 'w'), encoding='utf-8')
    else:
        df = pd.read_csv(constants.RESULTS_FULL_PATH, encoding='utf-8')
        # for _, value in OUTPUTS.iteritems():
        #         df[value[0]] = None
        #         df[value[1]] = None
    return df


def save_data(df):
    full_result_out = open(constants.RESULTS_FULL_PATH, 'w')
    df.to_csv(full_result_out, encoding='utf-8', columns=['result_id', 'decidim_category_id', 'decidim_scope_id', 'parent_id', 'external_id', 'start_date', 'end_date',  'decidim_accountability_status_id', 'progress', 'proposal_ids', 'title_ca',  'description_ca', 'title_es', 'description_es', 'Title Result ES',  'Title Result CA', 'Description Result ES', 'Description Result CA', 'Title Result ES_EN', 'Title Result ES_CA', 'Title Result CA_EN', 'Title Result CA_ES', 'Description Result CA_EN', 'Description Result CA_ES', 'Description Result ES_EN', 'Description Result ES_CA'])


def partial_translation(df):
    description = {}
    df_rows = df.shape[0]
    start_row = df[df['Title Result ES_EN'].isnull() | df['Title Result ES_CA'].isnull() | df['Title Result CA_EN'].isnull() |
                    df['Title Result CA_ES'].isnull() | df['Description Result ES_EN'].isnull() | df['Description Result ES_CA'].isnull() |
                        df['Description Result CA_EN'].isnull() | df['Description Result CA_ES'].isnull()].index[0]
    try:
        while start_row < df_rows:
            sub_df = df.iloc[start_row:start_row + CHUNK_SIZE]
            start_row += CHUNK_SIZE
            for idx in range(0, 2):
                for col in COLUMNS:  # To Translate Title in the first iteration and Description in the second
                    to_translate = []
                    chr_count = 0
                    for index, row in sub_df.iterrows():
                        if pd.isnull(row[OUTPUTS[col][idx]]):
                            if not pd.isnull(row[col]):
                                chr_count += len(row[col])
                                if col == "description_ca":
                                    text = remove_html_tags(row[col])
                                    description[text] = row[col]
                                else:
                                    text = row[col]
                                if chr_count < 100000:
                                    to_translate.append(text)
                                else:
                                    translation = call_google_translate(to_translate, LANGUAGES_MAP[col][idx][0], LANGUAGES_MAP[col][idx][1])
                                    add_result(df, col, OUTPUTS[col][idx], translation, description)
                                    description = {}
                                    to_translate = [text]
                                    chr_count = 0
                    if to_translate:
                        translation = call_google_translate(to_translate, LANGUAGES_MAP[col][idx][0], LANGUAGES_MAP[col][idx][1])
                        add_result(df, col, OUTPUTS[col][idx], translation, description)
                        description = {}
            save_data(df)
    except Exception as e:
        traceback.print_exc()
        print "Unexpected error: {}".format(e.message)
        save_data(df)


def call_google_translate(to_translate, source, target):
    translation = None
    while translation is None:
        try:
            translation = translate(to_translate, source, target)
        except Exception as e:
            print e.message
            sleep(5)  # To avoid exceed user limit rate

    return translation


def add_result(df, input_col, output_col, translation, description):
    for result in translation:
        if df[input_col].isin([result['input']]).any():
            df.loc[df.loc[:, input_col] == result['input'], [output_col]] = result['translatedText']
        elif result['input'] in description:
            df.loc[df.loc[:, input_col] == description[result['input']], [output_col]] = result['translatedText']


def translate(to_translate, source, target):
    return translate_client.translate(
        to_translate,
        source_language=source,
        target_language=target,
        format_=FORMAT)


def remove_html_tags(text):
    return re.sub(re.compile('<.*?>'), '', text)

if __name__ == '__main__':
    partial_translation(load_data())
