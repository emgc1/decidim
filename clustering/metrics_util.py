import os

import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn import metrics
import pandas as pd
import copy
from clustering.cluster_purity import purity_score
from plot_dendrogram import plot_dendrogram, plot_metrics_by_clusters
os.chdir("/Users/esthergonzalez/TesisDecidim")
path = "final/output/"


def compute_scores(dist_matrix, labels, labels_true, type, category, metric):
    methods = ['complete', 'average']
    best_scores = {}
    scores = {}
    labels_true = np.asarray(labels_true)
    if category in ['ALL', 'All']:
        labels_true = pd.Categorical(labels_true).codes

    for method in methods:
        best_scores[method] = {
            'nmi': {},
            'silhoutte': {},
            'purity': {},
            'ari': {}
        }
        if not (method in ('ward', 'centroid') and metric == 'cosine'):
            Z = linkage(dist_matrix, method)
            scores[method] = {}
            step = 0.02
            for max_d in np.arange(Z[0][2], Z[len(Z)-1][2], step):
                cut = cut_tree(Z, height=max_d)
                labels_pred = np.concatenate(cut).ravel()
                number_clusters = len(set(labels_pred))
                if number_clusters < len(dist_matrix) and len(squareform(dist_matrix))-1 >= number_clusters >= 2:
                        silhouette = metrics.silhouette_score(squareform(dist_matrix), list(labels_pred), metric="precomputed")
                        purity = purity_score(copy.deepcopy(labels_true), copy.deepcopy(labels_pred))
                        nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
                        ari = metrics.adjusted_rand_score(labels_true, labels_pred)
                        add_best_scores(best_scores[method], method, max_d, nmi, silhouette, number_clusters, purity, ari)
                        add_all_scores(scores[method], max_d, nmi, silhouette, number_clusters, purity, ari)

    plot_metrics_by_clusters(scores, type, metric, category)
    to_latex = {
        "nmi": {
            "linkage": None,
            "cut": None,
            "clusters": None,
            "nmi": -99,
            "silhoutte": None,
            "purity": None,
            "ari": None
        },
        "silhoutte": {
            "linkage": None,
            "cut": None,
            "clusters": None,
            "nmi": None,
            "silhoutte": -99,
            "purity": None,
            "ari": None
        },
        "purity": {
            "linkage": None,
            "cut": None,
            "clusters": None,
            "nmi": None,
            "silhoutte": None,
            "purity": -99,
            "ari": None
        },
        "ari": {
            "linkage": None,
            "cut": None,
            "clusters": None,
            "nmi": None,
            "silhoutte": None,
            "purity": None,
            "ari": -99
        }
    }

    for method in methods:
        for key, val in best_scores[method].iteritems():
            if 'score' in val:
                plot_dendrogram(Z, labels, labels_true, type, category, method, val['distance'], metric, "{}_{}".format(key,val.get('score')))
                if to_latex[key][key] < val.get('score'):
                    to_latex[key]["nmi"] = val.get('nmi')
                    to_latex[key]["silhoutte"] = val.get('silhoutte')
                    to_latex[key]["purity"] = val.get('purity')
                    to_latex[key]["ari"] = val.get('ari')
                    to_latex[key]["clusters"] = val['clusters']
                    to_latex[key]["cut"] = val['distance']
                    to_latex[key]["linkage"] = method
                    to_latex[key][key] = val.get('score')


    if to_latex['nmi']['linkage'] == to_latex['silhoutte']['linkage'] == to_latex['ari']['linkage'] == to_latex['purity']['linkage']:
        print " \\multirow{{4}}{{*}}{{{}}}  & \\multirow{{4}}{{*}}{{{}}}  & \\multirow{{4}}{{*}}{{{}}} & {:.4f} & {} & \\textbf{{{:.4f}}} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
            type, str(metric).title(), str(to_latex['nmi']['linkage']).title(), to_latex['nmi']['cut'], to_latex['nmi']['clusters'], to_latex['nmi']['nmi'],  to_latex['nmi']['purity'], to_latex['nmi']['ari'], to_latex['nmi']['silhoutte'])
        print " & &  & {:.4f} & {} & {:.4f} & \\textbf{{{:.4f}}} & {:.4f} & {:.4f} \\\\".format(to_latex['purity']['cut'], to_latex['purity']['clusters'], to_latex['purity']['nmi'], to_latex['purity']['purity'], to_latex['purity']['ari'],  to_latex['purity']['silhoutte'])
        print " & &  & {:.4f} & {} & {:.4f} & {:.4f} & \\textbf{{{:.4f}}} & {:.4f} \\\\".format(to_latex['ari']['cut'], to_latex['ari']['clusters'], to_latex['ari']['nmi'], to_latex['ari']['purity'], to_latex['ari']['ari'],  to_latex['ari']['silhoutte'])
        print " & &  & {:.4f} & {} & {:.4f} &  {:.4f} & {:.4f} & \\textbf{{{:.4f}}} \\\\".format(to_latex['silhoutte']['cut'], to_latex['silhoutte']['clusters'], to_latex['silhoutte']['nmi'], to_latex['silhoutte']['purity'], to_latex['silhoutte']['ari'],  to_latex['silhoutte']['silhoutte'])
        print "\\cline{1-9}"
    else:
        print "\\multirow{{4}}{{*}}{{{}}}  & \\multirow{{4}}{{*}}{{{}}}  & {} & {:.4f} & {} & \\textbf{{{:.4f}}} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
            type, str(metric).title(), str(to_latex['nmi']['linkage']).title(), to_latex['nmi']['cut'], to_latex['nmi']['clusters'], to_latex['nmi']['nmi'],  to_latex['nmi']['purity'], to_latex['nmi']['ari'], to_latex['nmi']['silhoutte'])
        print " & & {} & {:.4f} & {} & {:.4f} & \\textbf{{{:.4f}}}&  {:.4f} & {:.4f} \\\\".format(to_latex['purity']['linkage'], to_latex['purity']['cut'], to_latex['purity']['clusters'], to_latex['purity']['nmi'], to_latex['purity']['purity'], to_latex['purity']['ari'],  to_latex['purity']['silhoutte'])
        print " & & {} & {:.4f} & {} & {:.4f} &  {:.4f} & \\textbf{{{:.4f}}} & {:.4f}  \\\\".format(to_latex['ari']['linkage'], to_latex['ari']['cut'], to_latex['ari']['clusters'], to_latex['ari']['nmi'], to_latex['ari']['purity'], to_latex['ari']['ari'],  to_latex['ari']['silhoutte'])
        print " & & {} & {:.4f} & {} & {:.4f} &  {:.4f} & {:.4f} & \\textbf{{{:.4f}}} \\\\".format(to_latex['silhoutte']['linkage'], to_latex['silhoutte']['cut'], to_latex['silhoutte']['clusters'], to_latex['silhoutte']['nmi'], to_latex['silhoutte']['purity'], to_latex['silhoutte']['ari'],  to_latex['silhoutte']['silhoutte'])
        print "\\cline{1-9}"

    if type == "Word2Vec":
        print "\\hline"

    ## if constants.SUBCATEGORY in constants.CATEGORIES[constants.]:
    ##     main_category = constants.CATEGORIES[category][constants.SUBCATEGORY][constants.ENGLISH]
    ## pd.DataFrame.from_dict(best_scores, orient='index').to_csv('{}/{}/scores/{}_{}_{}_best_scores.csv'.format(path, main_category,category, str(type).lower(), str(metric).lower()))
    ## pd.DataFrame.from_dict(scores, orient='index').to_csv('{}/{}/scores/{}_{}_{}_all_scores.csv'.format(path, main_category,category, str(type).lower(), str(metric).lower()))

    # else:
    # pd.DataFrame.from_dict(best_scores, orient='index').to_csv('{}/{}/scores/{}_{}_best_scores.csv'.format(path, category, str(type).lower(), str(metric).lower()))
    # pd.DataFrame.from_dict(scores, orient='index').to_csv('{}/{}/scores/{}_{}_all_scores.csv'.format(path, category, str(type).lower(), str(metric).lower()))


def add_all_scores(scores, distance, nmi, silhouette, clusters, pur, ari):
    if 0 not in scores:
        scores[0] = {
            'distances': [],
            'scores': [],
            'name': 'Silhouette',
            'clusters': [],
            'dist_sc_cl': []
        }
    scores[0]['distances'].append(distance)
    scores[0]['scores'].append(silhouette)
    scores[0]['clusters'].append(clusters)
    scores[0]['dist_sc_cl'].append("{} - {} - {}".format(distance, silhouette, clusters))
    if 1 not in scores:
        scores[1] = {
            'distances': [],
            'scores': [],
            'name': 'NMI',
            'clusters': [],
            'dist_sc_cl': []
        }
    scores[1]['distances'].append(distance)
    scores[1]['scores'].append(nmi)
    scores[1]['clusters'].append(clusters)
    scores[1]['dist_sc_cl'].append("{} - {} - {}".format(distance, nmi, clusters))
    if 2 not in scores:
        scores[2] = {
            'distances': [],
            'scores': [],
            'name': 'Purity',
            'clusters': [],
            'dist_sc_cl': []
        }
    scores[2]['distances'].append(distance)
    scores[2]['scores'].append(pur)
    scores[2]['clusters'].append(clusters)
    scores[2]['dist_sc_cl'].append("{} - {} - {}".format(distance, nmi, clusters))

    if 3 not in scores:
        scores[3] = {
            'distances': [],
            'scores': [],
            'name': 'ARI',
            'clusters': [],
            'dist_sc_cl': []
        }
    scores[3]['distances'].append(distance)
    scores[3]['scores'].append(ari)
    scores[3]['clusters'].append(clusters)
    scores[3]['dist_sc_cl'].append("{} - {} - {}".format(distance, nmi, clusters))


def add_best_scores(max_dicts, method, distance, nmi, silhoutte, clusters, purity, ari):
    if clusters > 1:
        if nmi > max_dicts.get('nmi', {}).get('score', -999):
                max_dicts['nmi']['score'] = nmi
                max_dicts['nmi']['method'] = method
                max_dicts['nmi']['distance'] = distance
                max_dicts['nmi']['clusters'] = clusters
                max_dicts['nmi']['silhoutte'] = silhoutte
                max_dicts['nmi']['purity'] = purity
                max_dicts['nmi']['ari'] = ari
        if silhoutte > max_dicts.get('silhoutte', {}).get('score', -999):
                max_dicts['silhoutte']['score'] = silhoutte
                max_dicts['silhoutte']['method'] = method
                max_dicts['silhoutte']['distance'] = distance
                max_dicts['silhoutte']['clusters'] = clusters
                max_dicts['silhoutte']['nmi'] = nmi
                max_dicts['silhoutte']['purity'] = purity
                max_dicts['silhoutte']['ari'] = ari
        if ari > max_dicts.get('ari', {}).get('score', -999):
                max_dicts['ari']['score'] = ari
                max_dicts['ari']['method'] = method
                max_dicts['ari']['distance'] = distance
                max_dicts['ari']['clusters'] = clusters
                max_dicts['ari']['nmi'] = nmi
                max_dicts['ari']['purity'] = purity
                max_dicts['ari']['silhoutte'] = silhoutte
        if purity > max_dicts.get('purity', {}).get('score', -999):
                max_dicts['purity']['score'] = purity
                max_dicts['purity']['method'] = method
                max_dicts['purity']['distance'] = distance
                max_dicts['purity']['clusters'] = clusters
                max_dicts['purity']['nmi'] = nmi
                max_dicts['purity']['ari'] = ari
                max_dicts['purity']['silhoutte'] = silhoutte
