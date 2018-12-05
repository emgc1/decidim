import constants
import pandas as pd

en_columns = ['id', 'title', 'body', 'lang', 'title_en', 'body_en']


def get_proposals():
    proposals = pd.read_csv(constants.PROPOSAL_PATH, sep=',', header=0)
    proposals_en = pd.read_csv(constants.PROPOSALS_PARTIAL_EN_PATH, sep='\t', header=0)
    result = proposals.join(proposals_en.set_index('id'), on='Proposal ID')
    result.columns
    for idx, row in result.iterrows():
        if row['title'].strip().replace('\n', ' ').replace('\t', ' ') != row['Title'].strip().replace('\n', ' ').replace('\t', ' ') :
            print "{} Different titles \n  {} \n  {}".format(row['Proposal ID'], row['title'].strip().replace('\n', ' ').replace('\t', ' ') , row['Title'].strip().replace('\n', ' ').replace('\t', ' ') )
        if row['body'].strip().replace('\n', ' ').replace('\t', ' ') != row['Description'].strip().replace('\n', ' ').replace('\t', ' ') :
            print "{} Different descriptions \n  -{}-- \n  -{}--".format(row['Proposal ID'], row['body'].strip().replace('\n', ' ').replace('\t', ' ') , row['Description'].strip().replace('\n', ' ').replace('\t', ' ') )

    result.to_csv(open(constants.PROPOSALS_EN_PATH, 'w'), encoding='utf-8')

if __name__ == '__main__':
    get_proposals()