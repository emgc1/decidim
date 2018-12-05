# -*- coding: utf-8 -*-

import os

TITLE_COLUMN = "title_en"
BODY_COLUMN = "body_en"

ROOT_PATH = os.path.dirname(os.path.abspath(__file__)) + "/data"
PROPOSAL_PATH = ROOT_PATH + "/data/proposals.csv"
RESULTS_PATH = ROOT_PATH + "/data/results.csv"
RESULTS_FULL_PATH = ROOT_PATH + "/data/results_full.csv"
PROPOSALS_SAMPLE_PATH = ROOT_PATH + "/data/proposals_sample.csv"
PROPOSALS_SAMPLE_DISTANCES_PATH = ROOT_PATH + "/data/proposals_sample_distances.csv"
PROPOSALS_PARTIAL_EN_PATH = ROOT_PATH + "/data/proposals-valdivia-v2_detected_EN.csv"

# English path
PROPOSALS_EN_SAMPLE_PATH = ROOT_PATH + "/data_en/proposals_en_sample.csv"
PROPOSALS_EN_PATH = ROOT_PATH + "/data_en/proposals_en_full_result.csv"
STEM_PROPOSALS_PATH = ROOT_PATH + "/data_en/proposals_text_stem.csv"
LEM_PROPOSALS_PATH = ROOT_PATH + "/data_en/proposals_text_lem.csv"
ORIGINAL_PROPOSALS_PATH = ROOT_PATH + "/data_en/proposals_text_original.csv"

DENDROGRAMS_OUTPUT = ROOT_PATH + "/final/output/"


#########################
ROOT_PATH = os.path.dirname(os.path.abspath(__file__)) + "/data/final"
PROPOSALS_WITH_RESULT_PATH = ROOT_PATH + "/proposals_with_result.csv"

##Categories
NAME='name'
PROPOSALS_PATH='proposals_path'
ENGLISH = 'english'
PROPOSALS_RESULT_PATH='proposals_result_path'
JUSTICIA_GLOBAL = 'Justícia global'
TRANSICION_ECOLOGICA= 'Transició ecològica'
ECONOMIA_PLURAL = 'Economia plural'
BON_VIURE = 'Bon viure'
BON_GOVERN = 'Bon govern'
SUBCATEGORY='Subcategory'
CATEGORY="Category"
DEVELOPMENT_PROXIMITY = 'Proximity development and economy'
NEW_LEADERSHIP = 'Un nou lideratge públic'
SUSTAINABLE_TOURISM = 'Turisme sostenible'
QUALITY_EMPLOYMENT = 'Ocupació de qualitat'
COOPERATIVE_ECONOMY = 'Economia cooperativa, social i solidària'


CATEGORIES = {
    'ALL':{
        PROPOSALS_PATH: ROOT_PATH + "/proposals_with_result.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH + "/proposals_with_result.csv",
        ENGLISH: "All",
        'plot': 'small'
    },
    'All':{
        PROPOSALS_PATH: ROOT_PATH + "/proposals_with_result.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH + "/proposals_with_result.csv",
        ENGLISH: "All",
        'plot': 'small'
    },
    JUSTICIA_GLOBAL: {
        PROPOSALS_PATH:ROOT_PATH + "/proposals_justicia_global.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH + "/proposals_result_justicia_global.csv",
        ENGLISH: "Global Justice",
        'plot': 'small'

    },
    "Global Justice": {
        PROPOSALS_PATH:ROOT_PATH + "/proposals_justicia_global.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH + "/proposals_result_justicia_global.csv",
        ENGLISH: "Global Justice",
        'plot': 'small'

    },
    TRANSICION_ECOLOGICA: {
        PROPOSALS_PATH:ROOT_PATH + "/proposals_transicion_ecologica.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH + "/proposals_result_transicion_ecologica.csv",
        ENGLISH: "Ecological Transition",
        'plot': 'medium'
    },
    ECONOMIA_PLURAL: {
        PROPOSALS_PATH:ROOT_PATH + "/proposals_economia_plural.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH +  "/proposals_result_economia_plural.csv",
        ENGLISH: "Plural Economy",
        'plot': 'medium'
    },
    BON_VIURE: {
        PROPOSALS_PATH:ROOT_PATH + "/proposals_bon_govern.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH +  "/proposals_result_bon_govern.csv",
        ENGLISH: "Good Living",
        'plot': 'medium'
    },
    BON_GOVERN: {
        PROPOSALS_PATH:ROOT_PATH + "/proposals_bon_viure.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH +  "/proposals_result_bon_viure.csv",
        ENGLISH: "Good Governance",
        'plot': 'medium'
    },
    DEVELOPMENT_PROXIMITY: {
        SUBCATEGORY: True,
        CATEGORY: ECONOMIA_PLURAL,
        PROPOSALS_PATH: ROOT_PATH + "/proposals_economia_plural.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH + "/proposals_result_economia_plural.csv",
        ENGLISH: "Proximity development and economy"
    },
    COOPERATIVE_ECONOMY: {
        SUBCATEGORY: True,
        CATEGORY: ECONOMIA_PLURAL,
        PROPOSALS_PATH: ROOT_PATH + "/proposals_economia_plural.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH + "/proposals_result_economia_plural.csv",
        ENGLISH: "Cooperative and social economy"
    },
    QUALITY_EMPLOYMENT: {
        SUBCATEGORY: True,
        CATEGORY: ECONOMIA_PLURAL,
        PROPOSALS_PATH: ROOT_PATH + "/proposals_economia_plural.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH + "/proposals_result_economia_plural.csv",
        ENGLISH: "Quality employment"
    },
    SUSTAINABLE_TOURISM: {
        SUBCATEGORY: True,
        CATEGORY: ECONOMIA_PLURAL,
        PROPOSALS_PATH: ROOT_PATH + "/proposals_economia_plural.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH + "/proposals_result_economia_plural.csv",
        ENGLISH: "Sustainable tourism"
    },
    NEW_LEADERSHIP: {
        SUBCATEGORY: True,
        CATEGORY: ECONOMIA_PLURAL,
        PROPOSALS_PATH: ROOT_PATH + "/proposals_economia_plural.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH + "/proposals_result_economia_plural.csv",
        ENGLISH: "A new public leadership"
    },
    'A new public leadership': {
        SUBCATEGORY: True,
        CATEGORY: ECONOMIA_PLURAL,
        PROPOSALS_PATH: ROOT_PATH + "/proposals_economia_plural.csv",
        PROPOSALS_RESULT_PATH: ROOT_PATH + "/proposals_result_economia_plural.csv",
        ENGLISH: "A new public leadership"
    }
}

subcategories = {
    # 'Acció comunitària' : 'Community action',
    # 'Un nou lideratge públic': 'A new public leadership',
    'Ciutat d’acollida': 'Host city',
    # 'Eficiència i professionalitat': 'Efficiency and professionalism',
    'Esports': 'Sports',
    'Sanitat i salut': 'Health',
    'Govern transparent  i rendició de comptes': 'Transparent government',
    'Cultura': 'Culture'
    # ,
    # 'Turisme sostenible': 'Sustainable tourism'
}
