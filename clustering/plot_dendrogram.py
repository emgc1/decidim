import sys

import matplotlib
from scipy.cluster.hierarchy import dendrogram
import constants
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# matplotlib.use('TkAgg')
# sys.setrecursionlimit(10000)


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    annotate = kwargs.pop('annotate', True)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        if annotate:
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if max_d > y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center', fontsize=7)
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def plot_dendrogram_small_ds(Z, labels, labels_true,  type, category, method, max_d, metric, filename):
    fig = plt.figure()
    fig.set_size_inches(50, 15)
    small_fontsize = 15
    medium_fontsize = 25

    plt.ylabel('{} Distance'.format(str(metric).title()), fontsize=medium_fontsize)
    if category=="ALL" or category=='All':
        fig.set_size_inches(13, 12)
        plt.xlabel('Category - Proposal ID', labelpad=15, fontsize=medium_fontsize)
        plt.title("Clustering of Proposals\n {} Linkage".format(str(method).title()), fontsize=medium_fontsize)
    else:
        plt.xlabel('Proposal - Result Ids', fontsize=medium_fontsize, labelpad=15)
        plt.title("Clustering of {} Proposals\n {} - {} Linkage".format(category, type, str(method).title()), fontsize=medium_fontsize)

    fancy_dendrogram(
        Z,
        show_leaf_counts=False,
        leaf_rotation=90.,
        show_contracted=True,
        labels=labels,
        max_d=max_d,
        color_threshold=max_d,
        above_threshold_color='grey'
    )
    plt.xticks(fontsize=small_fontsize)
    plt.yticks(fontsize=small_fontsize)
    # fig.subplots_adjust(bottom=0.3)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def plot_dendrogram_medium_ds(Z, labels, labels_true,  type, category, method, max_d, metric, filename):
    fig = plt.figure()
    fig.set_size_inches(30, 20)
    fig.tight_layout()
    if type=="LSA_5" or type=="LSA_10":
        plt.title("Clustering of {} Proposals\n LSA - {} Linkage".format(category,  str(method).title()), fontsize=25)
    else:
        plt.title("Clustering of {} Proposals\n {} - {} Linkage".format(category, type, str(method).title()), fontsize=25)
    plt.xlabel('Proposal - Result Ids', fontsize=20, labelpad=15)
    plt.ylabel('{} Distance'.format(str(metric).title()), fontsize=20)

    fancy_dendrogram(
        Z,
        show_leaf_counts=False,
        leaf_rotation=90.,
        show_contracted=True,
        labels=labels,
        max_d=max_d,
        color_threshold=max_d,
        above_threshold_color='grey'
    )
    fig.savefig(filename)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    # fig.tight_layout()

    plt.close()

def plot_dendrogram_big_ds(Z, labels, labels_true,  type, category, method, max_d, metric, filename):
    fig = plt.figure()
    fig.set_size_inches(120, 50)
    fig.tight_layout()
    plt.title("Clustering of {} Proposals\n {} - {} Linkage".format(category, type, str(method).title()), fontsize=25)
    plt.xlabel('Proposal - Result Ids', fontsize=20, labelpad=15)
    plt.ylabel('{} Distance'.format(str(metric).title()), fontsize=20)
    fig.subplots_adjust(bottom=0.2)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=13)
    fancy_dendrogram(
        Z,
        show_leaf_counts=False,
        leaf_rotation=90.,
        leaf_font_size=5,
        show_contracted=True,
        labels=labels,
        max_d=max_d,
        color_threshold=max_d,
        above_threshold_color='grey',
        annotate = False
    )
    fig.savefig(filename + '.pdf')
    plt.close()


def plot_dendrogram(Z, labels, labels_true,  type, category, method, max_d, metric, scores):
    if category in constants.CATEGORIES and constants.SUBCATEGORY in constants.CATEGORIES[category]:
        filename = constants.DENDROGRAMS_OUTPUT+"/{}/{}/{}_{}_{}_{}_{}_{}_{}.png".format(constants.CATEGORIES[category][constants.CATEGORY], category, str(type).lower(), str(metric).lower(), category, metric, method, max_d, scores)
    else:
        filename = constants.DENDROGRAMS_OUTPUT+"/{}/{}_{}_{}_{}_{}_{}_{}.png".format(category, str(type).lower(), str(metric).lower(), category, metric, method, max_d, scores)

    if category in constants.CATEGORIES and constants.CATEGORIES[category]['plot'] == 'small':
        plot_dendrogram_small_ds(Z, labels, labels_true,  type, category, method, max_d, metric, filename)
    if category in constants.CATEGORIES and constants.CATEGORIES[category]['plot'] == 'medium':
        plot_dendrogram_medium_ds(Z, labels, labels_true,  type, category, method, max_d, metric, filename)
    if category in constants.CATEGORIES and constants.CATEGORIES[category]['plot'] ==  'big':
        plot_dendrogram_big_ds(Z, labels, labels_true,  type, category, method, max_d, metric, filename)


def plot_metrics_by_clusters(scores, type, metric, category):
    cmap = matplotlib.cm.get_cmap('Dark2')
    if metric == 'cosine':
        row = 1
        col = 2
        gs = gridspec.GridSpec(3, 1)
    else:
        gs = gridspec.GridSpec(4, 1)
        row = 2
        col = 2

    for x in ['distances', 'clusters']:
        fig, axes = plt.subplots(row, col, sharex=True, sharey=True)
        if x == 'distances':
            if metric=='cosine':
                fig.text(0.5, 0.13, 'Cut Threshold ({} distance)'.format(str(metric).title()), ha='center')
            else:
                fig.text(0.5, 0.1, 'Cut Threshold ({} distance)'.format(str(metric).title()), ha='center')
            fig.suptitle('Clustering Score Value vs Cut Threshold ({})'.format(type))
        else:
            if metric=='cosine':
                fig.text(0.5, 0.13, 'Number of Clusters'.format(str(metric).title()), ha='center')
            else:
                fig.text(0.5, 0.1, 'Number of Clusters'.format(str(metric).title()), ha='center')
            fig.suptitle('Clustering Score Value vs Number of Clusters ({})'.format(type))

        fig.text(0.06, 0.5, 'Score Value', va='center', rotation='vertical')

        idm = 0
        for key, value in scores.iteritems():
            plt.subplot(gs[idm, :])
            idm += 1
            method = key
            for id in range(0, len(value)):
                plt.plot(value[id][x], value[id]['scores'], '.:', label=value[id]['name'], color=cmap(id))
            plt.grid(True)
            plt.title('Linkage {}'.format(method), fontsize=10)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.subplots_adjust(hspace=0.5)


        if metric == 'cosine':
            plt.legend(loc='upper center', bbox_to_anchor=(0.48, -0.6), fancybox=True, shadow=True, ncol=len(value))
            # fig.set_size_inches(6.5, 4.8)
        else:
            plt.subplots_adjust(hspace=0.65)
            plt.legend(loc='upper center', bbox_to_anchor=(0.48, -0.6), fancybox=True, shadow=True, ncol=len(value))
            # fig.set_size_inches(6.5, 6)

        fig.subplots_adjust(bottom=0.2)

        if category in constants.CATEGORIES and constants.SUBCATEGORY in constants.CATEGORIES[category]:
            fig.savefig("{}/{}/{}/{}_score_{}_{}_{}.png".format(constants.DENDROGRAMS_OUTPUT, constants.CATEGORIES[category][constants.CATEGORY], category, x, str(type).lower(), str(metric).lower(), method))
        else:
            fig.savefig("{}/{}/{}_score_{}_{}_{}.png".format(constants.DENDROGRAMS_OUTPUT, category, x, str(type).lower(), str(metric).lower(), method))
        # fig.tight_layout()

        plt.close()