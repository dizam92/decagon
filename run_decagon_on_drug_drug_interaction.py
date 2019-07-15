from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
__author__ = 'maoss2'

from operator import itemgetter
from itertools import combinations
import time
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics
from collections import defaultdict
from itertools import product
from sklearn.model_selection import train_test_split

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator, EdgeMinibatchIteratorNewSplit
from decagon.utility import rank_metrics, preprocessing

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(0)


def get_accuracy_scores(feed_dict, placeholders, sess, opt,
                        minibatch, adj_mats_orig, edges_pos,
                        edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i, j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders


# TENSORFLOW VARIABLES
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 150

val_test_size = 0.05


# LOADERS
def load_drug_bank_combo_side_effect_file(fichier):
    combo_se = pd.read_csv(fichier)
    if 'Unnamed: 0' in combo_se.columns:
        combo_se.drop(['Unnamed: 0'], axis=1, inplace=True)
    edges_and_se = list(zip(combo_se['Drug 1'].values, combo_se['Drug 2'].values, combo_se['ddi type'].values))
    edges_to_be_separated = [el for el in edges_and_se if el[2].find(';') != -1]
    edges_and_se = [el for el in edges_and_se if el[2].find(';') == -1]
    for edges_to_be_process in edges_to_be_separated:
        tmp = edges_to_be_process[2].split(';')
        new_edge_1 = (edges_to_be_process[0], edges_to_be_process[1], tmp[0])
        new_edge_2 = (edges_to_be_process[0], edges_to_be_process[1], tmp[1])
        edges_and_se.append(new_edge_1)
        edges_and_se.append(new_edge_2)
    combo_to_drugs_ids = {'{}_{}'.format(drug_edge[0], drug_edge[1]): drug_edge[:2] for drug_edge in edges_and_se}
    combo_to_side_effects = {'{}_{}'.format(drug_edge[0], drug_edge[1]): drug_edge[2] for drug_edge in edges_and_se}
    return combo_to_drugs_ids, combo_to_side_effects


# I do a loader for the decagon file cause that the same of two side but with one side effect per line
def load_decagon_combo_side_effect_file(fichier):
    combo_to_side_effects = defaultdict()
    combo_to_side_effects_names = defaultdict()
    combo_se = pd.read_csv(fichier)
    edges_and_se = list(zip(combo_se['STITCH 1'].values,
                            combo_se['STITCH 2'].values,
                            combo_se['Polypharmacy Side Effect'].values,
                            combo_se['Side Effect Name'].values))
    combo_to_drugs_ids = {'{}_{}'.format(drug_edge[0], drug_edge[1]): drug_edge[:2] for drug_edge in edges_and_se}
    for drug_edge in edges_and_se:
        if '{}_{}'.format(drug_edge[0], drug_edge[1]) not in combo_to_side_effects.keys():
            combo_to_side_effects['{}_{}'.format(drug_edge[0], drug_edge[1])] = [drug_edge[2]]
            combo_to_side_effects_names['{}_{}'.format(drug_edge[0], drug_edge[1])] = [drug_edge[3]]
        else:
            combo_to_side_effects['{}_{}'.format(drug_edge[0], drug_edge[1])].append(drug_edge[2])
            combo_to_side_effects_names['{}_{}'.format(drug_edge[0], drug_edge[1])].append(drug_edge[3])
    side_effects_ids_to_names = {drug_edge[2]: drug_edge[3] for drug_edge in edges_and_se}
    # combo_to_side_effects = {'{}_{}'.format(drug_edge[0], drug_edge[1]): drug_edge[2] for drug_edge in edges_and_se}
    # combo_to_side_effects_names = {'{}_{}'.format(drug_edge[0], drug_edge[1]): drug_edge[3] for drug_edge in edges_and_se}
    return combo_to_drugs_ids, combo_to_side_effects, combo_to_side_effects_names, side_effects_ids_to_names


# Load the db targets file and the twosides target file
def load_file_targets_id(fichier):
    drug_target_id = pd.read_csv(fichier)
    edges_drug_ids_and_names_and_target_ids = list(zip(drug_target_id['drug_id'].values,
                                                       drug_target_id['drug_name'].values,
                                                       drug_target_id['targets_id'].values))
    edges_to_be_separeted = [el for el in edges_drug_ids_and_names_and_target_ids
                             if type(el[2]) != float and el[2].find(';') != -1]
    edges_drug_ids_and_names_and_target_ids = [el for el in edges_drug_ids_and_names_and_target_ids
                                               if type(el[2]) != float and el[2].find(';') == -1]
    for edges_to_be_process in edges_to_be_separeted:
        tmp = edges_to_be_process[2].split(';')
        for i in range(len(tmp)):
            edges_drug_ids_and_names_and_target_ids.append((edges_to_be_process[0], edges_to_be_process[1], tmp[i]))
    drugs_id_to_drugs_name = {drug_edge[0]: drug_edge[1] for drug_edge in edges_drug_ids_and_names_and_target_ids}
    drugs_id_to_targets_id = defaultdict()  # Only difference it's a dictionary of list cause we can have a drug
    # that have many targets
    for drug_edge in edges_drug_ids_and_names_and_target_ids:
        if drug_edge[0] not in drugs_id_to_targets_id.keys():
            drugs_id_to_targets_id[drug_edge[0]] = [drug_edge[2]]
        else:
            drugs_id_to_targets_id[drug_edge[0]].append(drug_edge[2])
    return drugs_id_to_targets_id, drugs_id_to_drugs_name


# Load the decagon targets file
def load_decagon_file_targets_id(fichier):
    drug_target_id = pd.read_csv(fichier)
    edges_drug_ids_and_target_ids = list(zip(drug_target_id['STITCH'].values,
                                             drug_target_id['Gene'].values))
    drugs_id_to_targets_id = defaultdict()  # Only difference it's a dictionary of list cause we can have a drug
    # that have many targets
    for drug_edge in edges_drug_ids_and_target_ids:
        if drug_edge[0] not in drugs_id_to_targets_id.keys():
            drugs_id_to_targets_id[drug_edge[0]] = [drug_edge[1]]
        else:
            drugs_id_to_targets_id[drug_edge[0]].append(drug_edge[1])
    return drugs_id_to_targets_id


# Returns networkx graph of the PPI network and a dictionary that maps each gene ID to a number (For Two sides)
def load_genes_genes_interactions(fichier):
    genes_genes_interactions = pd.read_csv(fichier)
    edges = list(zip(genes_genes_interactions['Gene 1'].values, genes_genes_interactions['Gene 2'].values))
    nodes = set([u for e in edges for u in e])
    print('Edges: %d' % len(edges))
    print('Nodes: %d' % len(nodes))
    net = nx.Graph()
    net.add_edges_from(edges)
    net.remove_nodes_from(nx.isolates(net))
    net.remove_edges_from(net.selfloop_edges())
    node2idx = {node: i for i, node in enumerate(net.nodes())}
    return net, node2idx


# loader DDI define by Rogia for loading the ddi file: load DB and TS and the Split files generated
# by the train_test_Split function
def load_ddis_combinations(fname, header=True, dataset_name="twosides"):
    fn = open(fname)
    if header:
        fn.readline()
    combo2se = defaultdict(list)
    for line in fn:
        content = line.split(",")
        if dataset_name not in ["twosides", "split"]:
            content = content[1:]
        drug1 = content[0]
        drug2 = content[1]
        se = content[-1].strip("\n").split(";")
        combo2se[(drug1, drug2)] = list(set(se))
    return combo2se


def train_test_valid_split_3():
    interactions = load_ddis_combinations(fname='/home/maoss2/PycharmProjects/decagon/polypharmacy/twosides/twosides-combo.csv',
                                          header=True,
                                          dataset_name='twosides')
    drugs = list(set([x1 for (x1, _) in interactions] + [x2 for (_, x2) in interactions]))

    np.random.seed(42)
    train_idx, test_idx = train_test_split(drugs, test_size=0.1, shuffle=True)
    train_idx, valid_idx = train_test_split(train_idx, test_size=0.15, shuffle=True)

    train = set(product(train_idx, repeat=2))
    valid = set(product(train_idx, valid_idx)).union(set(product(valid_idx, train_idx)))
    test = set(product(train_idx, test_idx)).union(set(product(test_idx, train_idx)))

    train = set(interactions.keys()).intersection(train)
    valid = set(interactions.keys()).intersection(valid)
    test = set(interactions.keys()).intersection(test)

    print('len train', len(list(train)))
    print('len test', len(list(test)))
    print('len valid', len(list(valid)))
    print("len region grise", len(interactions) - (len(train) + len(test) + len(valid)))

    combo_to_drugs_ids, combo_to_side_effects, combo_to_side_effects_names, side_effects_ids_to_names =\
        load_decagon_combo_side_effect_file(fichier='/home/maoss2/PycharmProjects/decagon/polypharmacy/bio-decagon-combo.csv')

    train_keys = ['{}_{}'.format(el[0], el[1]) for el in train]
    test_keys = ['{}_{}'.format(el[0], el[1]) for el in test]
    valid_keys = ['{}_{}'.format(el[0], el[1]) for el in valid]

    combo_to_drugs_ids_train = {cle: combo_to_drugs_ids[cle] for cle in train_keys if cle in combo_to_drugs_ids.keys()}
    combo_to_drugs_ids_test = {cle: combo_to_drugs_ids[cle] for cle in test_keys if cle in combo_to_drugs_ids.keys()}
    combo_to_drugs_ids_valid = {cle: combo_to_drugs_ids[cle] for cle in valid_keys if cle in combo_to_drugs_ids.keys()}

    # combo_to_side_effects = {cle: combo_to_side_effects[cle] for cle in train_keys
    #                          if cle in combo_to_side_effects.keys()}
    # combo_to_side_effects_names = {cle: combo_to_side_effects_names[cle] for cle in train_keys
    #                                if cle in combo_to_side_effects_names.keys()}
    # side_effects_ids_to_names = {cle: side_effects_ids_to_names[cle] for cle in train_keys
    #                              if cle in side_effects_ids_to_names.keys()}

    return combo_to_drugs_ids_train, combo_to_drugs_ids_test, combo_to_drugs_ids_valid


def main_execution(combo_file='./polypharmacy/bio-decagon-combo.csv',
                   targets_file='./polypharmacy/bio-decagon-targets.csv',
                   genes_genes_file='./polypharmacy/bio-decagon-ppi.csv',
                   new_train_test_split=False):
    print('Load Combo to Side Effects')
    if combo_file.find('decagon') != -1:
        combo_to_drugs_ids, combo_to_side_effects, combo_to_side_effects_names, side_effects_ids_to_names = \
            load_decagon_combo_side_effect_file(fichier=combo_file)
        print('Load drugs to targets')
        drugs_id_to_targets_id = load_decagon_file_targets_id(fichier=targets_file)
    else:
        combo_to_drugs_ids, combo_to_side_effects = load_drug_bank_combo_side_effect_file(fichier=combo_file)
        print('Load drugs to targets')
        drugs_id_to_targets_id, drugs_id_to_drugs_name = load_file_targets_id(fichier=targets_file)

    print('Load genes to genes (targets) interactions net')
    genes_genes_net, genes_node_to_idx = load_genes_genes_interactions(fichier=genes_genes_file)

    print('Build genes-genes adjacency matrix')
    genes_adj = nx.adjacency_matrix(genes_genes_net)
    genes_degrees = np.array(genes_adj.sum(axis=0)).squeeze()

    if new_train_test_split:
        print('Load the new train test validation split')
        combo_to_drugs_ids_train, combo_to_drugs_ids_test, combo_to_drugs_ids_valid = train_test_valid_split_3()
        drug_nodes_train = set([u for e in combo_to_drugs_ids_train.values() for u in e])
        drug_nodes_test = set([u for e in combo_to_drugs_ids_test.values() for u in e])
        drug_nodes_valid = set([u for e in combo_to_drugs_ids_valid.values() for u in e])

    print('Build drugs-drugs matrix representation')
    drug_nodes = set([u for e in combo_to_drugs_ids.values() for u in e])
    n_drugs = len(drug_nodes)
    relation_types = set([r for se in combo_to_side_effects.values() for r in se])
    drugs_nodes_to_idx = {node: i for i, node in enumerate(drug_nodes)}

    print('Build general drugs-drugs matrix representation')
    drug_drug_adj_list = []  # matrice d'adjacence de chaque drug_drug
    for i, el in enumerate(relation_types):  # pour chaque side effect
        mat = np.zeros((n_drugs, n_drugs))
        for d1, d2 in combinations(list(drug_nodes), 2):
            temp_cle = '{}_{}'.format(d1, d2)
            if temp_cle in combo_to_side_effects.keys():
                if el in combo_to_side_effects[temp_cle]:
                    # list of list on check si le s.e apparait au moins une fois dans la liste
                    mat[drugs_nodes_to_idx[d1], drugs_nodes_to_idx[d2]] = \
                        mat[drugs_nodes_to_idx[d2], drugs_nodes_to_idx[d1]] = 1.
                    # Inscrire une interaction
        drug_drug_adj_list.append(sp.csr_matrix(mat))
    drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]

    if new_train_test_split:
        print('Build train drugs-drugs matrix representation')
        drug_drug_adj_list_train = []  # matrice d'adjacence de chaque drug_drug
        for i, el in enumerate(relation_types):  # pour chaque side effect
            mat = np.zeros((n_drugs, n_drugs))
            for d1, d2 in combinations(list(drug_nodes_train), 2):
                temp_cle = '{}_{}'.format(d1, d2)
                if temp_cle in combo_to_side_effects.keys():
                    if el in combo_to_side_effects[temp_cle]:
                        # list of list on check si le s.e apparait au moins une fois dans la liste
                        mat[drugs_nodes_to_idx[d1], drugs_nodes_to_idx[d2]] = \
                            mat[drugs_nodes_to_idx[d2], drugs_nodes_to_idx[d1]] = 1.
                     # Inscrire une interaction
            drug_drug_adj_list_train.append(sp.csr_matrix(mat))
        drug_degrees_list_train = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list_train]

        print('Build test drugs-drugs matrix representation')
        drug_drug_adj_list_test = []  # matrice d'adjacence de chaque drug_drug
        for i, el in enumerate(relation_types):  # pour chaque side effect
            mat = np.zeros((n_drugs, n_drugs))
            for d1, d2 in combinations(list(drug_nodes_test), 2):
                temp_cle = '{}_{}'.format(d1, d2)
                if temp_cle in combo_to_side_effects.keys():
                    if el in combo_to_side_effects[temp_cle]:
                        # list of list on check si le s.e apparait au moins une fois dans la liste
                        mat[drugs_nodes_to_idx[d1], drugs_nodes_to_idx[d2]] = \
                            mat[drugs_nodes_to_idx[d2], drugs_nodes_to_idx[d1]] = 1.
                    # Inscrire une interaction
            drug_drug_adj_list_test.append(sp.csr_matrix(mat))
        drug_degrees_list_test = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list_test]

        print('Build valid drugs-drugs matrix representation')
        drug_drug_adj_list_valid = []  # matrice d'adjacence de chaque drug_drug
        for i, el in enumerate(relation_types):  # pour chaque side effect
            mat = np.zeros((n_drugs, n_drugs))
            for d1, d2 in combinations(list(drug_nodes_valid), 2):
                temp_cle = '{}_{}'.format(d1, d2)
                if temp_cle in combo_to_side_effects.keys():
                    if el in combo_to_side_effects[temp_cle]:
                        # list of list on check si le s.e apparait au moins une fois dans la liste
                        mat[drugs_nodes_to_idx[d1], drugs_nodes_to_idx[d2]] = \
                            mat[drugs_nodes_to_idx[d2], drugs_nodes_to_idx[d1]] = 1.
                    # Inscrire une interaction
            drug_drug_adj_list_valid.append(sp.csr_matrix(mat))
        drug_degrees_list_valid = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list_valid]

    print('Build general genes-drugs matrix representation')
    genes_nodes = set([gene_node for gene_node in genes_node_to_idx.keys()])
    n_genes = len(genes_nodes)
    mat = np.zeros((n_genes, n_drugs))
    for drug in drug_nodes:
        if drug in drugs_id_to_targets_id.keys():
            for target in drugs_id_to_targets_id[drug]:
                if target in genes_node_to_idx.keys():
                    mat[genes_node_to_idx[target], drugs_nodes_to_idx[drug]] = 1.
    genes_drugs_adj = sp.csr_matrix(mat)
    drugs_genes_adj = genes_drugs_adj.transpose(copy=True)

    if new_train_test_split:
        print('Build train genes-drugs matrix representation')
        for drug in drug_nodes_train:
            if drug in drugs_id_to_targets_id.keys():
                for target in drugs_id_to_targets_id[drug]:
                    if target in genes_node_to_idx.keys():
                        mat[genes_node_to_idx[target], drugs_nodes_to_idx[drug]] = 1.
        genes_drugs_adj_train = sp.csr_matrix(mat)
        drugs_genes_adj_train = genes_drugs_adj_train.transpose(copy=True)

        print('Build test genes-drugs matrix representation')
        for drug in drug_nodes_test:
            if drug in drugs_id_to_targets_id.keys():
                for target in drugs_id_to_targets_id[drug]:
                    if target in genes_node_to_idx.keys():
                        mat[genes_node_to_idx[target], drugs_nodes_to_idx[drug]] = 1.
        genes_drugs_adj_test = sp.csr_matrix(mat)
        drugs_genes_adj_test = genes_drugs_adj_test.transpose(copy=True)

        print('Build valid genes-drugs matrix representation')
        for drug in drug_nodes_valid:
            if drug in drugs_id_to_targets_id.keys():
                for target in drugs_id_to_targets_id[drug]:
                    if target in genes_node_to_idx.keys():
                        mat[genes_node_to_idx[target], drugs_nodes_to_idx[drug]] = 1.
        genes_drugs_adj_valid = sp.csr_matrix(mat)
        drugs_genes_adj_valid = genes_drugs_adj_valid.transpose(copy=True)

    print('Build general Adjacency matrix data representation')
    adj_mats_orig = {
        (0, 0): [genes_adj, genes_adj.transpose(copy=True)],
        (0, 1): [genes_drugs_adj],
        (1, 0): [drugs_genes_adj],
        (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
    }

    if new_train_test_split:
        print('Build train Adjacency matrix data representation')
        adj_mats_orig_train = {
            (0, 0): [genes_adj, genes_adj.transpose(copy=True)],
            (0, 1): [genes_drugs_adj_train],
            (1, 0): [drugs_genes_adj_train],
            (1, 1): drug_drug_adj_list_train + [x.transpose(copy=True) for x in drug_drug_adj_list_train],
        }

        print('Build test Adjacency matrix data representation')
        adj_mats_orig_test = {
            (0, 0): [genes_adj, genes_adj.transpose(copy=True)],
            (0, 1): [genes_drugs_adj_test],
            (1, 0): [drugs_genes_adj_test],
            (1, 1): drug_drug_adj_list_test + [x.transpose(copy=True) for x in drug_drug_adj_list_test],
        }

        print('Build valid Adjacency matrix data representation')
        adj_mats_orig_valid = {
            (0, 0): [genes_adj, genes_adj.transpose(copy=True)],
            (0, 1): [genes_drugs_adj_valid],
            (1, 0): [drugs_genes_adj_valid],
            (1, 1): drug_drug_adj_list_valid + [x.transpose(copy=True) for x in drug_drug_adj_list_valid],
        }

    degrees = {
        0: [genes_degrees, genes_degrees],
        1: drug_degrees_list + drug_degrees_list,
    }

    print('featureless (genes)')
    gene_feat = sp.identity(n_genes)
    gene_nonzero_feat, gene_num_feat = gene_feat.shape
    gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

    print('features (drugs)')
    drug_feat = sp.identity(n_drugs)
    drug_nonzero_feat, drug_num_feat = drug_feat.shape
    drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

    print('Features data representation')
    num_feat = {
        0: gene_num_feat,
        1: drug_num_feat,
    }
    nonzero_feat = {
        0: gene_nonzero_feat,
        1: drug_nonzero_feat,
    }
    feat = {
        0: gene_feat,
        1: drug_feat,
    }

    edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
    edge_type2decoder = {
        (0, 0): 'bilinear',
        (0, 1): 'bilinear',
        (1, 0): 'bilinear',
        (1, 1): 'dedicom',
    }

    edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
    num_edge_types = sum(edge_types.values())
    print("Edge types:", "%d" % num_edge_types)

    print("Defining placeholders")
    placeholders = construct_placeholders(edge_types)

    ###########################################################
    #
    # Create minibatch iterator, model and optimizer
    #
    ###########################################################

    if new_train_test_split:
        print("Create minibatch iterator")
        minibatch = EdgeMinibatchIteratorNewSplit(
            adj_mats=adj_mats_orig,
            adj_mats_train=adj_mats_orig_train,
            adj_mats_test=adj_mats_orig_test,
            adj_mats_valid=adj_mats_orig_valid,
            feat=feat,
            edge_types=edge_types,
            batch_size=FLAGS.batch_size,
            val_test_size=val_test_size
        )
    else:
        print("Create minibatch iterator")
        minibatch = EdgeMinibatchIterator(
            adj_mats=adj_mats_orig,
            feat=feat,
            edge_types=edge_types,
            batch_size=FLAGS.batch_size,
            val_test_size=val_test_size
        )

    print("Create model")
    model = DecagonModel(
        placeholders=placeholders,
        num_feat=num_feat,
        nonzero_feat=nonzero_feat,
        edge_types=edge_types,
        decoders=edge_type2decoder,
    )

    print("Create optimizer")
    with tf.name_scope('optimizer'):
        opt = DecagonOptimizer(
            embeddings=model.embeddings,
            latent_inters=model.latent_inters,
            latent_varies=model.latent_varies,
            degrees=degrees,
            edge_types=edge_types,
            edge_type2dim=edge_type2dim,
            placeholders=placeholders,
            batch_size=FLAGS.batch_size,
            margin=FLAGS.max_margin
        )

    print("Initialize session")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict = {}

    ###########################################################
    #
    # Train model
    #
    ###########################################################

    print("Train model")
    for epoch in range(FLAGS.epochs):

        minibatch.shuffle()
        itr = 0
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
            feed_dict = minibatch.update_feed_dict(
                feed_dict=feed_dict,
                dropout=FLAGS.dropout,
                placeholders=placeholders)

            t = time.time()

            # Training step: run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
            train_cost = outs[1]
            batch_edge_type = outs[2]

            if itr % PRINT_PROGRESS_EVERY == 0:
                val_auc, val_auprc, val_apk = get_accuracy_scores(feed_dict, placeholders, sess, opt,
                                                                  minibatch, adj_mats_orig,
                                                                  minibatch.val_edges, minibatch.val_edges_false,
                                                                  minibatch.idx2edge_type[
                                                                      minibatch.current_edge_type_idx])

                print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                      "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))

            itr += 1

    print("Optimization finished!")

    for et in range(num_edge_types):
        roc_score, auprc_score, apk_score = get_accuracy_scores(feed_dict, placeholders, sess, opt,
                                                                minibatch, adj_mats_orig,
                                                                minibatch.test_edges,
                                                                minibatch.test_edges_false, minibatch.idx2edge_type[et])
        print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
        print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
        print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
        print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
        print()


if __name__ == '__main__':
    # main_execution(combo_file='./polypharmacy/bio-decagon-combo.csv',
    #                targets_file='./polypharmacy/bio-decagon-targets.csv',
    #                genes_genes_file='./polypharmacy/bio-decagon-ppi.csv',
    #                new_train_test_split=True)

    main_execution(combo_file='/home/maoss2/PycharmProjects/decagon/polypharmacy/bio-decagon-combo.csv',
                   targets_file='/home/maoss2/PycharmProjects/decagon/polypharmacy/bio-decagon-targets.csv',
                   genes_genes_file='/home/maoss2/PycharmProjects/decagon/polypharmacy/bio-decagon-ppi.csv',
                   new_train_test_split=True)