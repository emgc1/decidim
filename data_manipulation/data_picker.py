import constants
import pandas as pd
import random
import numpy as np

result_columns = ['result_id','decidim_category_id', 'decidim_scope_id', 'proposal_ids', 'title_ca', 'description_ca', 'title_es',
               'description_es', 'Title Result ES', 'Title Result CA', 'Description Result ES', 'Description Result CA',
               'Title Result ES_EN', 'Title Result ES_CA', 'Title Result CA_EN', 'Title Result CA_ES', 'Description Result CA_EN',
                'Description Result CA_ES', 'Description Result ES_EN', 'Description Result ES_CA']

categories_columns = ['Proposal ID', 'result_id', 'decidim_category_id', 'Category','Subcategory', 'decidim_scope_id', 'Scope',
              'title_en', 'body_en','proposal_ids','Description Result ES_EN']


def get_actuations_by_id(result_ids):
    result_df = pd.read_csv(constants.RESULTS_FULL_PATH, encoding='utf-8')
    sample_df = pd.DataFrame()
    for id in result_ids:
        sample_df = sample_df.append(result_df.loc[result_df.loc[:, 'result_id'] == id])
    return sample_df


def get_actuations_by_proposals_id(proposals_df):
    proposals_id = proposals['Proposal ID'].tolist()
    result_df = pd.read_csv(constants.RESULTS_FULL_PATH, encoding='utf-8')
    proposals_sample_df = pd.DataFrame()
    proposals_result_sample_df = pd.DataFrame()
    for idx, result in result_df.iterrows():
        if isinstance(result['proposal_ids'], basestring) or not np.isnan(result['proposal_ids']):
            ids =[int(x) for x in  result['proposal_ids'].split(';')]
            proposals_by_result = set(proposals_id).intersection(set(ids))
            for pid in proposals_by_result:
                proposal_row = proposals_df.loc[proposals_df.loc[:, 'Proposal ID'] == pid]
                proposal_result_row = pd.concat([proposal_row.iloc[0], result], axis=0).to_frame().T
                proposal_result_row.drop(['Description Result ES_CA', 'Description Result ES_EN', 'Description Result CA_ES', 'Title Result CA_ES', 'Title Result CA_EN', 'Title Result ES_CA', 'Description Result CA', 'Description Result ES', 'Title Result CA', 'Title Result ES'], axis=1)
                proposals_sample_df = proposals_sample_df.append(proposal_row)
                proposals_result_sample_df = proposals_result_sample_df.append(proposal_result_row)
    return proposals_sample_df, proposals_result_sample_df


def get_actuations_categories():
    df_results = pd.read_csv(constants.RESULTS_FULL_PATH, encoding='utf-8')
    df_proposals = pd.read_csv(constants.PROPOSALS_EN_PATH, encoding='utf-8')
    df_proposals_results = pd.DataFrame()
    count = 0
    for idx, result in df_results.iterrows():
        if isinstance(result['proposal_ids'], basestring) or not np.isnan(result['proposal_ids']):
            ids = result['proposal_ids'].split(';')
            if len(ids) > 1:
                scope = None
                found = False
                for id in ids:
                    row = df_proposals.loc[df_proposals.loc[:, 'Proposal ID'] == int(id)]
                    if not scope:
                        scope = row['Scope'].iloc[0]
                    if not found and scope != row['Scope'].iloc[0]:
                        print "Scope mismatch result id {}".format(result['result_id'])
                        found = True
                        count += 1
                    for col in categories_columns:
                        if col in df_results.columns:
                            row[col] = result[col]
                    df_proposals_results = df_proposals_results.append(row)
    print count
    return df_proposals_results


def get_actuacions(quantity):
    df = pd.read_csv(constants.RESULTS_FULL_PATH, encoding='utf-8')
    sample_df = pd.DataFrame()
    for x in range(0, quantity):
        row = df.iloc[random.randint(0, df.shape[0])]
        sample_df = sample_df.append(row)
    return sample_df


def get_actuations_with_proposals():
    df_results = pd.read_csv(constants.RESULTS_FULL_PATH, encoding='utf-8')
    df_proposals = pd.read_csv(constants.PROPOSALS_EN_PATH, encoding='utf-8')
    proposals_result_sample_df = pd.DataFrame()
    count = 0
    for idx, result in df_results.iterrows():
        if isinstance(result['proposal_ids'], basestring) or not np.isnan(result['proposal_ids']):
            ids = result['proposal_ids'].split(';')
            for id in ids:
                proposal_row = df_proposals.loc[df_proposals.loc[:, 'Proposal ID'] == int(id)]
                proposal_result_row = pd.concat([proposal_row.iloc[0], result], axis=0).to_frame().T
                proposal_result_row.drop(['Description Result ES_CA', 'Description Result ES_EN', 'Description Result CA_ES', 'Title Result CA_ES', 'Title Result CA_EN', 'Title Result ES_CA', 'Description Result CA', 'Description Result ES', 'Title Result CA', 'Title Result ES'], axis=1)
                proposals_result_sample_df = proposals_result_sample_df.append(proposal_result_row)
        if count%100 == 0:
            save_actuations_with_proposals(proposals_result_sample_df)
    return proposals_result_sample_df


def get_proposals(act_df):
    df = pd.read_csv(constants.PROPOSALS_EN_PATH, encoding='utf-8')
    proposals_sample = pd.DataFrame()
    for idx, result in act_df.iterrows():
        if isinstance(result['proposal_ids'], basestring) or not np.isnan(result['proposal_ids']):
            ids = result['proposal_ids'].split(';')
            if len(ids) >= 2:
                print ids
                for id in ids:
                    row = df.loc[df.loc[:, 'Proposal ID'] == int(id)].copy()
                    for col in result_columns:
                        row[col] = result[col]
                    proposals_sample = proposals_sample.append(row)
    return proposals_sample


def get_proposals_by_ids(ids):
    df = pd.read_csv(constants.PROPOSALS_EN_PATH, encoding='utf-8')
    sample_df = pd.DataFrame()
    for id in ids:
        row = df.loc[df.loc[:, 'Proposal ID'] == id]
        sample_df = sample_df.append(row)
    return sample_df


def get_proposals_by_category(category):
    df = pd.read_csv(constants.PROPOSALS_EN_PATH, encoding='utf-8')
    if category:
        return df.loc[df['Category'] == category.decode("utf-8")]
    else:
        return df


def save_sample(df):
    full_result_out = open(constants.PROPOSALS_SAMPLE_PATH, 'w')
    df.to_csv(full_result_out, encoding='utf-8', columns=['Proposal ID','Origin','Scope','District','Category','Subcategory',
                                                          'Author ID','Author Username','Comments','Created at','Title',
                                                          'Title ES','Title CA','Title_ES_CAT','Title_CA_ES','Title_ES_EN',
                                                          'Title_CA_EN','Description','Description ES','Description CA',
                                                          'Description_ES_CAT','Description_CA_ES','Description_ES_EN',
                                                          'Description_CA_EN','Status','URL','Votes','result_id','decidim_category_id',
                                                          'decidim_scope_id','proposal_ids','title_ca','description_ca','title_es',
                                                          'description_es','Title Result ES','Title Result CA','Description Result ES',
                                                          'Description Result CA','Title Result ES_EN','Title Result ES_CA',
                                                          'Title Result CA_EN','Title Result CA_ES','Description Result CA_EN',
                                                          'Description Result CA_ES','Title Result ES_EN','Description Result ES_EN','Description Result ES_CA']
              )


def save_proposals_result_category(df):
    full_result_out = open(constants.PROPOSALS_EN_CATEGORIES_PATH, 'w')
    df.to_csv(full_result_out, encoding='utf-8', columns=['Proposal ID', 'result_id', 'decidim_category_id', 'Category','Subcategory', 'decidim_scope_id', 'Scope',
                                                          'title_en', 'body_en','proposal_ids','Description Result ES_EN'])


def save_sample_en(df):
    full_result_out = open(constants.PROPOSALS_EN_SAMPLE_PATH, 'w')
    df.to_csv(full_result_out, encoding='utf-8')


def save_sample(proposals_sample_df, proposals_result_sample_df, category):
    proposals_out = open(constants.CATEGORIES[category][constants.PROPOSALS_PATH], 'w')
    result_out = open(constants.CATEGORIES[category][constants.PROPOSALS_RESULT_PATH], 'w')
    proposals_sample_df.to_csv(proposals_out, encoding='utf-8')
    proposals_result_sample_df.to_csv(result_out, encoding='utf-8')


def save_actuations_with_proposals(proposals_df):
    proposals_out = open(constants.PROPOSALS_WITH_RESULT_PATH, 'w')
    proposals_df.to_csv(proposals_out, encoding='utf-8')


if __name__ == '__main__':
    # proposals_sample = get_proposals(get_actuacions(15))
    # save_sample(proposals_sample)
    # proposals_sample = get_proposals_by_ids(ids)

    # proposals_sample = get_proposals(get_actuations_by_id(result_ids))
    # save_sample_en(proposals_sample)

    # # save_proposals_result_category(get_actuations_categories())
    # get_actuations_categories()

    for key, item in constants.CATEGORIES.iteritems():
        proposals = get_proposals_by_category(key)
        proposals_sample_df, proposals_result_sample_df = get_actuations_by_proposals_id(proposals)
        save_sample(proposals_sample_df, proposals_result_sample_df, key)
    # get_actuations_with_proposals()