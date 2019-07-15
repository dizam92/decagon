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
import scipy.sparse as sp
from sklearn import metrics

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(0)


def get_accuracy_scores(feed_dict, placeholders, sess, opt, minibatch, adj_mats_orig, edges_pos, edges_neg, edge_type):
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
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

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
        for i, j in edge_types for k in range(edge_types[i,j])})
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
    edges_to_be_separeted = [el for el in edges_and_se if el[2].find(';') != -1]
    edges_and_se = [el for el in edges_and_se if el[2].find(';') == -1]
    for edges_to_be_process in edges_to_be_separeted:
        tmp = edges_to_be_process[2].split(';')
        new_edge_1 = (edges_to_be_process[0], edges_to_be_process[1], tmp[0])
        new_edge_2 = (edges_to_be_process[0], edges_to_be_process[1], tmp[1])
        edges_and_se.append(new_edge_1)
        edges_and_se.append(new_edge_2)
    combo_to_drugs_ids = {'{}_{}'.format(drug_edge[0], drug_edge[1]): drug_edge[:2] for drug_edge in edges_and_se}
    combo_to_side_effects = {'{}_{}'.format(drug_edge[0], drug_edge[1]): drug_edge[2] for drug_edge in edges_and_se}
    return combo_to_drugs_ids, combo_to_side_effects

def main_execution():
    combo_to_drugs_ids, combo_to_side_effects = load_drug_bank_combo_side_effect_file(fichier='polypharmacy/drugbank/drugbank-combo.csv')
    nodes = set([u for e in combo_to_drugs_ids.values() for u in e])
    n_drugs = len(nodes)
    relation_types = set([r for r in combo_to_side_effects.values()])
    n_drugdrug_rel_types = len(relation_types)
    drugs_to_positions_in_matrices_dict = {node: i for i, node in enumerate(nodes)}

    drug_drug_adj_list = []  # matrice d'adjacence de chaque drug_drug
    for i, el in enumerate(relation_types):  # pour chaque side effect
        mat = np.zeros((n_drugs, n_drugs))
        for d1, d2 in combinations(list(nodes), 2):
            temp_cle = '{}_{}'.format(d1, d2)
            if temp_cle in combo_to_side_effects.keys():
                if combo_to_side_effects[temp_cle] == el:
                    # chaque fois on a une réelle s.e entre les 2 drogues dans la matrice
                    mat[drugs_to_positions_in_matrices_dict[d1], drugs_to_positions_in_matrices_dict[d2]] = \
                        mat[drugs_to_positions_in_matrices_dict[d2], drugs_to_positions_in_matrices_dict[d1]] = 1.
                    # Inscrire une interaction
        drug_drug_adj_list.append(sp.csr_matrix(mat))
    drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]

    adj_mats_orig = {
        (0, 0): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
    }
    degrees = {
        0: drug_degrees_list + drug_degrees_list,
    }

    # features (drugs)
    drug_feat = sp.identity(n_drugs)
    drug_nonzero_feat, drug_num_feat = drug_feat.shape
    drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

    # data representation
    num_feat = {
        0: drug_num_feat,
    }
    nonzero_feat = {
        0: drug_nonzero_feat,
    }
    feat = {
        0: drug_feat,
    }

    edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
    edge_type2decoder = {
        (0, 0): 'dedicom',
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
                                                                  minibatch.idx2edge_type[minibatch.current_edge_type_idx])

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
    main_execution()