# coding: UTF-8
import pandas as pd
import numpy as np
import scipy as sp
import lightgbm as lgb 
import jieba
from sklearn.model_selection import train_test_split, StratifiedKFold
import feather
import keras
import tensorflow as tf
import codecs
import gc
from multiprocessing import Pool
import contextlib
import pickle
import os
from sklearn.metrics import roc_auc_score

from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.activations import *
from keras.optimizers import *
import warnings
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer

from joblib import delayed, Parallel, dump, load

import distance
from fuzzywuzzy import fuzz

import os
import gc
import random

from gensim.models import Word2Vec
    
seed = 12

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

question = pd.read_csv('../input/question.csv')
question = question[['qid','words', 'chars']]
train = pd.read_csv('../input/train.csv')
train_r = pd.read_csv('../input/train_remix3.csv')
test  = pd.read_csv('../input/test.csv')
test_r  = pd.read_csv('../input/test_fix.csv')

train = pd.merge(train, question, left_on=['q1'], right_on=['qid'], how='left').drop(['qid'], axis=1)
train = pd.merge(train, question, left_on=['q2'], right_on=['qid'], how='left').drop(['qid'], axis=1)

train_r = pd.merge(train_r, question, left_on=['q1'], right_on=['qid'], how='left').drop(['qid'], axis=1)
train_r = pd.merge(train_r, question, left_on=['q2'], right_on=['qid'], how='left').drop(['qid'], axis=1)

test = pd.merge(test, question, left_on=['q1'], right_on=['qid'], how='left').drop(['qid'], axis=1)
test = pd.merge(test, question, left_on=['q2'], right_on=['qid'], how='left').drop(['qid'], axis=1)

test_r = pd.merge(test_r, question, left_on=['q1'], right_on=['qid'], how='left').drop(['qid'], axis=1)
test_r = pd.merge(test_r, question, left_on=['q2'], right_on=['qid'], how='left').drop(['qid'], axis=1)


if not os.path.exists("../input/word_embed.own%s" %seed):
    corpus = np.concatenate([train.words_x.values, train.words_y.values,  test.words_x.values, test.words_y.values], axis=0)
    seg_corpus = [_.split() for _ in corpus]
    print(len(seg_corpus))
    word2vec = Word2Vec(seg_corpus, size=200, min_count=2, sg=1, hs=0, negative=10, iter=10, workers=42, window=5, seed = seed)
    word2vec.wv.save_word2vec_format("../input/word_embed.own%s" %seed)
else:
    pass
    
if not os.path.exists("../input/char_embed.own%s" %seed):
    from gensim.models import Word2Vec
    corpus = np.concatenate([train.chars_x.values, train.chars_y.values,  test.chars_x.values, test.chars_y.values], axis=0)
    seg_corpus = [_.split() for _ in corpus]
    print(len(seg_corpus))
    word2vec = Word2Vec(seg_corpus, size=200, min_count=2, sg=1, hs=0, negative=10, iter=10, workers=40, window=6, seed = seed)
    word2vec.wv.save_word2vec_format("../input/char_embed.own%s" %seed)
else:
    pass

MAX_NB_WORDS = 300000                                             
                                                                                                                        
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)                     
tokenizer.fit_on_texts(question['words'])                         
train_q1_word_seq = tokenizer.texts_to_sequences(train['words_x'])
train_q2_word_seq = tokenizer.texts_to_sequences(train['words_y'])
test_q1_word_seq = tokenizer.texts_to_sequences(test['words_x'])
test_q2_word_seq = tokenizer.texts_to_sequences(test['words_y'])
train_r_q1_word_seq = tokenizer.texts_to_sequences(train_r['words_x'])
train_r_q2_word_seq = tokenizer.texts_to_sequences(train_r['words_y'])
test_r_q1_word_seq = tokenizer.texts_to_sequences(test_r['words_x'])
test_r_q2_word_seq = tokenizer.texts_to_sequences(test_r['words_y'])

word_index = tokenizer.word_index                              
embeddings_index = {}                                          
with open('../input/word_embed.own%s' %seed,'r') as f:                 
    for i in f:                                                                                                   
        values = i.split(' ')                                                                                      
        if len(values) == 2: continue                                                                              
        word = str(values[0])                                                                                      
        embedding = np.asarray(values[1:],dtype='float')                                                           
        embeddings_index[word] = embedding
print('word embedding',len(embeddings_index))                                                                           
                                       
EMBEDDING_DIM = 200   
nb_words = min(MAX_NB_WORDS,len(word_index))                                                                  
                                                           
word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(str(word).upper())
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector

MAX_WORD_SEQUENCE_LENGTH = 20
train_q1_word_seq  = pad_sequences(train_q1_word_seq, maxlen=MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0)
train_q2_word_seq  = pad_sequences(train_q2_word_seq, maxlen=MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0) 

test_q1_word_seq   = pad_sequences(test_q1_word_seq, maxlen=MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0)
test_q2_word_seq   = pad_sequences(test_q2_word_seq, maxlen=MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0)

train_r_q1_word_seq  = pad_sequences(train_r_q1_word_seq, maxlen=MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0)
train_r_q2_word_seq  = pad_sequences(train_r_q2_word_seq, maxlen=MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0)

test_r_q1_word_seq   = pad_sequences(test_r_q1_word_seq, maxlen=MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0)
test_r_q2_word_seq   = pad_sequences(test_r_q2_word_seq, maxlen=MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0)

y = train.label.values
y_r = train_r.label.values

MAX_NB_WORDS = 300000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(question['chars'])

train_q1_char_seq = tokenizer.texts_to_sequences(train['chars_x'])
train_q2_char_seq = tokenizer.texts_to_sequences(train['chars_y'])

test_q1_char_seq = tokenizer.texts_to_sequences(test['chars_x'])
test_q2_char_seq = tokenizer.texts_to_sequences(test['chars_y'])

train_r_q1_char_seq = tokenizer.texts_to_sequences(train_r['chars_x'])
train_r_q2_char_seq = tokenizer.texts_to_sequences(train_r['chars_y'])

test_r_q1_char_seq = tokenizer.texts_to_sequences(test_r['chars_x'])
test_r_q2_char_seq = tokenizer.texts_to_sequences(test_r['chars_y'])

word_index = tokenizer.word_index

embeddings_index = {}
with open('../input/char_embed.own%s' %seed,'r') as f:
    for i in f:
        values = i.split(' ')
        if len(values) == 2: continue
        word = str(values[0])
        embedding = np.asarray(values[1:],dtype='float')
        embeddings_index[word] = embedding
print('char embedding',len(embeddings_index))

EMBEDDING_DIM = 200
nb_words = min(MAX_NB_WORDS, len(word_index))

char_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(str(word).upper())
    if embedding_vector is not None:
        char_embedding_matrix[i] = embedding_vector

MAX_CHAR_SEQUENCE_LENGTH = 35
train_q1_char_seq  = pad_sequences(train_q1_char_seq, maxlen=MAX_CHAR_SEQUENCE_LENGTH, truncating='post', value=0)
train_q2_char_seq  = pad_sequences(train_q2_char_seq, maxlen=MAX_CHAR_SEQUENCE_LENGTH, truncating='post', value=0) 

test_q1_char_seq   = pad_sequences(test_q1_char_seq, maxlen=MAX_CHAR_SEQUENCE_LENGTH, truncating='post', value=0)
test_q2_char_seq   = pad_sequences(test_q2_char_seq, maxlen=MAX_CHAR_SEQUENCE_LENGTH, truncating='post', value=0)

train_r_q1_char_seq  = pad_sequences(train_r_q1_char_seq, maxlen=MAX_CHAR_SEQUENCE_LENGTH, truncating='post', value=0)
train_r_q2_char_seq  = pad_sequences(train_r_q2_char_seq, maxlen=MAX_CHAR_SEQUENCE_LENGTH, truncating='post', value=0)

test_r_q1_char_seq   = pad_sequences(test_r_q1_char_seq, maxlen=MAX_CHAR_SEQUENCE_LENGTH, truncating='post', value=0)
test_r_q2_char_seq   = pad_sequences(test_r_q2_char_seq, maxlen=MAX_CHAR_SEQUENCE_LENGTH, truncating='post', value=0)

if not os.path.exists("../cache/features_aug_%s.jl" %seed):
# if True:
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    import networkx as nx

    NB_CORES = 10

    def create_question_hash(train_df, test_df):
        train_qs = np.dstack([train_df["q1"], train_df["q2"]]).flatten()
        test_qs  = np.dstack([test_df["q1"], test_df["q2"]]).flatten()
        all_qs = np.append(train_qs, test_qs)
        all_qs = pd.DataFrame(all_qs)[0].drop_duplicates()
        all_qs.reset_index(inplace=True, drop=True)
        question_dict = pd.Series(all_qs.index.values, index=all_qs.values).to_dict()
        return question_dict


    def get_hash(df, hash_dict):
        df["qid1"] = df["q1"].map(hash_dict)
        df["qid2"] = df["q2"].map(hash_dict)
        return df.drop(["q1", "q2"], axis=1)


    def get_kcore_dict(df):
        g = nx.Graph()
        g.add_nodes_from(df.qid1)
        edges = list(df[["qid1", "qid2"]].to_records(index=False))
        g.add_edges_from(edges)
        g.remove_edges_from(g.selfloop_edges())

        print (len(g.nodes()))

        df_output = pd.DataFrame(data=list(g.nodes()), columns=["qid"])
        df_output["kcore"] = 0
        for k in range(2, NB_CORES + 1):
            ck = nx.k_core(g, k=k).nodes()
            print("kcore", k)
            df_output.ix[df_output.qid.isin(ck), "kcore"] = k
        return df_output.to_dict()["kcore"]


    def get_kcore_features(df, kcore_dict):
        df["kcore1"] = df["qid1"].apply(lambda x: kcore_dict[x])
        df["kcore2"] = df["qid2"].apply(lambda x: kcore_dict[x])
        return df


    def convert_to_minmax(df, col):
        sorted_features = np.sort(np.vstack([df[col + "1"], df[col + "2"]]).T)
        df["min_" + col] = sorted_features[:, 0]
        df["max_" + col] = sorted_features[:, 1]
        return df.drop([col + "1", col + "2"], axis=1)


    def get_neighbors(train_df, test_df):
        neighbors = defaultdict(set)
        for df in [train_df, test_df]:
            for q1, q2 in zip(df["qid1"], df["qid2"]):
                neighbors[q1].add(q2)
                neighbors[q2].add(q1)
        return neighbors


    def get_neighbor_features(df, neighbors):
        common_nc = df.apply(lambda x: len(neighbors[x.qid1].intersection(neighbors[x.qid2])), axis=1)
        min_nc = df.apply(lambda x: min(len(neighbors[x.qid1]), len(neighbors[x.qid2])), axis=1)
        df["common_neighbor_ratio"] = common_nc / min_nc
        df["common_neighbor_count"] = common_nc
        return df


    def get_freq_features(df, frequency_map):
        df["freq1"] = df["qid1"].map(lambda x: frequency_map[x])
        df["freq2"] = df["qid2"].map(lambda x: frequency_map[x])
        return df

    train_df = train[['q1', 'q2']]
    test_df  = test[['q1', 'q2']]
    train_r_df = train_r[['q1', 'q2']]
    test_r_df  = test_r[['q1', 'q2']]

    print("Hashing the questions...")
    question_dict = create_question_hash(train_df, test_df)
    train_df = get_hash(train_df, question_dict)
    test_df  = get_hash(test_df, question_dict)
    train_r_df = get_hash(train_r_df, question_dict)
    test_r_df  = get_hash(test_r_df, question_dict)
    print("Number of unique questions:", len(question_dict))

    print("Calculating kcore features...")
    all_df = pd.concat([train_df, test_df])
    kcore_dict = get_kcore_dict(all_df)
    train_df = get_kcore_features(train_df, kcore_dict)
    test_df  = get_kcore_features(test_df, kcore_dict)
    train_df = convert_to_minmax(train_df, "kcore")
    test_df  = convert_to_minmax(test_df, "kcore")
    train_r_df = get_kcore_features(train_r_df, kcore_dict)
    test_r_df  = get_kcore_features(test_r_df, kcore_dict)
    train_r_df = convert_to_minmax(train_r_df, "kcore")
    test_r_df  = convert_to_minmax(test_r_df, "kcore")

    print("Calculating common neighbor features...")
    neighbors = get_neighbors(train_df, test_df)
    train_df  = get_neighbor_features(train_df, neighbors)
    test_df   = get_neighbor_features(test_df, neighbors)
    train_r_df  = get_neighbor_features(train_r_df, neighbors)
    test_r_df   = get_neighbor_features(test_r_df, neighbors)

    print("Calculating frequency features...")
    frequency_map = dict(zip(*np.unique(np.vstack((all_df["qid1"], all_df["qid2"])), return_counts=True)))
    train_df      = get_freq_features(train_df, frequency_map)
    test_df = get_freq_features(test_df, frequency_map)
    train_df = convert_to_minmax(train_df, "freq")
    test_df  = convert_to_minmax(test_df, "freq")
    train_r_df      = get_freq_features(train_r_df, frequency_map)
    test_r_df = get_freq_features(test_r_df, frequency_map)
    train_r_df = convert_to_minmax(train_r_df, "freq")
    test_r_df  = convert_to_minmax(test_r_df, "freq")


    cols = ["min_kcore", "max_kcore", "common_neighbor_count", "common_neighbor_ratio", "min_freq", "max_freq", 'qid1', 'qid2']
    features_magic_train = train_df.loc[:, cols]
    features_magic_test  = test_df.loc[:, cols]
    features_magic_train_r = train_r_df.loc[:, cols]
    features_magic_test_r  = test_r_df.loc[:, cols]

    # coding: utf-8

    import networkx as nx
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import NMF

    path = '../cache/'

    G = nx.Graph()
    for q1, q2 in train[['q1', 'q2']].values:
        G.add_edge(q1,q2)   

    for q1, q2 in test[['q1', 'q2']].values:
        G.add_edge(q1,q2)       

    A = nx.adjacency_matrix(G)
    nmf = NMF(n_components=2, random_state=2018)
    adjacency_matrix_nmf = nmf.fit_transform(A)

    A = nx.incidence_matrix(G)
    nmf = NMF(n_components=2,random_state=2018)
    incidence_matrix_nmf = nmf.fit_transform(A)

    nodes = G.nodes()

    d = dict()
    for n,a,i in zip(nodes,adjacency_matrix_nmf,incidence_matrix_nmf):
        d[n] = np.concatenate([a,i])

    train_q1_decom = np.vstack(train['q1'].apply(lambda x:d[x]).values.tolist())
    train_q2_decom = np.vstack(train['q2'].apply(lambda x:d[x]).values.tolist())
    test_q1_decom  = np.vstack(test['q1'].apply(lambda x:d[x]).values.tolist())
    test_q2_decom  = np.vstack(test['q2'].apply(lambda x:d[x]).values.tolist())
    train_r_q1_decom = np.vstack(train_r['q1'].apply(lambda x:d[x]).values.tolist())
    train_r_q2_decom = np.vstack(train_r['q2'].apply(lambda x:d[x]).values.tolist())
    test_r_q1_decom  = np.vstack(test_r['q1'].apply(lambda x:d[x]).values.tolist())
    test_r_q2_decom  = np.vstack(test_r['q2'].apply(lambda x:d[x]).values.tolist())

    train_decom_diff  = pd.DataFrame(np.abs(train_q1_decom-train_q2_decom), columns=["decom_diff_%s" %i for i in range(4)])
    train_decom_angle = pd.DataFrame(train_q1_decom*train_q2_decom, columns=["decom_angle_%s" %i for i in range(4)])

    test_decom_diff  = pd.DataFrame(np.abs(test_q1_decom-test_q2_decom), columns=["decom_diff_%s" %i for i in range(4)])
    test_decom_angle = pd.DataFrame(test_q1_decom*test_q2_decom, columns=["decom_angle_%s" %i for i in range(4)])

    train_r_decom_diff  = pd.DataFrame(np.abs(train_r_q1_decom-train_r_q2_decom), columns=["decom_diff_%s" %i for i in range(4)])
    train_r_decom_angle = pd.DataFrame(train_r_q1_decom*train_r_q2_decom, columns=["decom_angle_%s" %i for i in range(4)])

    test_r_decom_diff  = pd.DataFrame(np.abs(test_r_q1_decom-test_r_q2_decom), columns=["decom_diff_%s" %i for i in range(4)])
    test_r_decom_angle = pd.DataFrame(test_r_q1_decom*test_r_q2_decom, columns=["decom_angle_%s" %i for i in range(4)])

    from csv import DictReader
    from math import exp, log, sqrt
    from random import random,shuffle
    import pickle
    import sys
    import random
    import networkx as nx


    G = nx.Graph()
    for q1, q2 in train[['q1', 'q2']].values:
        G.add_edge(q1,q2)   

    for q1, q2 in test[['q1', 'q2']].values:
        G.add_edge(q1,q2)    

    avg_degrees = nx.average_neighbor_degree(G)

    def hash_subgraph(q1, q2):
        q1_idf = avg_degrees.get(q1, 0)
        q2_idf = avg_degrees.get(q2, 0)
        qmax = max(q1_idf, q2_idf)
        qmin = min(q1_idf, q2_idf)
        qdiff = qmax - qmin
        qmean = 0.5*(q1_idf + q2_idf)
        return [qmax, qmin, qdiff, qmean]


    def extract_hash_subgraph_features(df):
        print("hash_subgraph features...")
        hash_subgraph_features = Parallel(n_jobs=40)(delayed(hash_subgraph)(x[0], x[1]) for x in df[['q1', 'q2']].values)
        df["hash_subgraph_qmax"]  = list(map(lambda x: x[0], hash_subgraph_features))
        df["hash_subgraph_qmin"]  = list(map(lambda x: x[1], hash_subgraph_features))
        df["hash_subgraph_qdiff"] = list(map(lambda x: x[2], hash_subgraph_features))
        df["hash_subgraph_qmean"]   = list(map(lambda x: x[3], hash_subgraph_features))
        del hash_subgraph_features; gc.collect()

    extract_hash_subgraph_features(train)
    extract_hash_subgraph_features(test)

    features_hash_subgraph_train = train[["hash_subgraph_qmax", "hash_subgraph_qmin", "hash_subgraph_qdiff", "hash_subgraph_qmean"]]
    features_hash_subgraph_test  = test[["hash_subgraph_qmax", "hash_subgraph_qmin", "hash_subgraph_qdiff", "hash_subgraph_qmean"]]

    extract_hash_subgraph_features(train_r)
    extract_hash_subgraph_features(test_r)

    features_hash_subgraph_train_r = train_r[["hash_subgraph_qmax", "hash_subgraph_qmin", "hash_subgraph_qdiff", "hash_subgraph_qmean"]]
    features_hash_subgraph_test_r  = test_r[["hash_subgraph_qmax", "hash_subgraph_qmin", "hash_subgraph_qdiff", "hash_subgraph_qmean"]]


    G = nx.Graph()
    for q1, q2 in train[['q1', 'q2']].values:
        G.add_edge(q1,q2)

    for q1, q2 in test[['q1', 'q2']].values:
        G.add_edge(q1,q2)

    avg_degrees = nx.average_neighbor_degree(G)

    def hash_subgraph(q1, q2):
        q1_idf = avg_degrees.get(q1, 0)
        q2_idf = avg_degrees.get(q2, 0)
        qmax = max(q1_idf, q2_idf)
        qmin = min(q1_idf, q2_idf)
        qdiff = qmax - qmin
        qmean = 0.5*(q1_idf + q2_idf)
        return [qmax, qmin, qdiff, qmean]


    def extract_hash_subgraph_features(df):
        print("hash_subgraph features...")
        hash_subgraph_features = Parallel(n_jobs=40)(delayed(hash_subgraph)(x[0], x[1]) for x in df[['q1', 'q2']].values)
        df["hash_subgraph_qmax"]  = list(map(lambda x: x[0], hash_subgraph_features))
        df["hash_subgraph_qmin"]  = list(map(lambda x: x[1], hash_subgraph_features))
        df["hash_subgraph_qdiff"] = list(map(lambda x: x[2], hash_subgraph_features))
        df["hash_subgraph_qmean"]   = list(map(lambda x: x[3], hash_subgraph_features))
        del hash_subgraph_features; gc.collect()

    extract_hash_subgraph_features(train)
    extract_hash_subgraph_features(test)

    features_hash_subgraph_train = train[["hash_subgraph_qmax", "hash_subgraph_qmin", "hash_subgraph_qdiff", "hash_subgraph_qmean"]]
    features_hash_subgraph_test  = test[["hash_subgraph_qmax", "hash_subgraph_qmin", "hash_subgraph_qdiff", "hash_subgraph_qmean"]]

    extract_hash_subgraph_features(train_r)
    extract_hash_subgraph_features(test_r)

    features_hash_subgraph_train_r = train_r[["hash_subgraph_qmax", "hash_subgraph_qmin", "hash_subgraph_qdiff", "hash_subgraph_qmean"]]
    features_hash_subgraph_test_r  = test_r[["hash_subgraph_qmax", "hash_subgraph_qmin", "hash_subgraph_qdiff", "hash_subgraph_qmean"]]

    import networkx as nx
    import numpy as np
    import pandas as pd


    G = nx.Graph()
    for q1, q2 in train[['q1', 'q2']].values:
        G.add_edge(q1,q2)   

    for q1, q2 in test[['q1', 'q2']].values:
        G.add_edge(q1,q2)  


    pr = nx.pagerank(G, alpha=0.9)

    def pagerank(q1, q2):
        pr1 = pr[q1] * 1e6
        pr2 = pr[q2] * 1e6
        return [max(pr1, pr2), min(pr1, pr2), (pr1 + pr2) / 2.]

    def extract_pagerank_features(df):
        print("pagerank features...")
        pagerank_features = Parallel(n_jobs=20)(delayed(pagerank)(x[0], x[1]) for x in df[['q1', 'q2']].values)
        df["max_pr"]  = list(map(lambda x: x[0], pagerank_features ))
        df["min_pr"]  = list(map(lambda x: x[1], pagerank_features ))
        df["mean_pr"] = list(map(lambda x: x[2], pagerank_features ))
        del pagerank_features; gc.collect()

    extract_pagerank_features(train)
    extract_pagerank_features(test)

    pagerank_features_train = train[["max_pr", "min_pr", "mean_pr"]]
    pagerank_features_test  = test[["max_pr", "min_pr", "mean_pr"]]

    extract_pagerank_features(train_r)
    extract_pagerank_features(test_r)

    pagerank_features_train_r = train_r[["max_pr", "min_pr", "mean_pr"]]
    pagerank_features_test_r  = test_r[["max_pr", "min_pr", "mean_pr"]]

    features_train = pd.concat([features_magic_train, train_decom_angle,  train_decom_diff, features_hash_subgraph_train, pagerank_features_train], axis=1).values
    features_test  = pd.concat([features_magic_test,  test_decom_angle,  test_decom_diff, features_hash_subgraph_test, pagerank_features_test], axis=1).values
    features_train_r = pd.concat([features_magic_train_r, train_r_decom_angle,  train_r_decom_diff, features_hash_subgraph_train_r, pagerank_features_train_r], axis=1).values
    features_test_r  = pd.concat([features_magic_test_r,  test_r_decom_angle,  test_r_decom_diff, features_hash_subgraph_test_r, pagerank_features_test_r], axis=1).values

    dump((features_train, features_test,features_train_r, features_test_r), "../cache/features_aug_%s.jl" %seed)
else:
    features_train, features_test,features_train_r, features_test_r = load( "../cache/features_aug_%s.jl" %seed)

for i in range(features_test.shape[0]):
    features_test_r[i] = features_test[test_r.loc[i,'oldindex']]

print features_test_r[:5]

def lstm_cross_char_add_feature():
    char_embedding_layer = Embedding(name="char_embedding",
                                     input_dim=char_embedding_matrix.shape[0],
                                     weights=[char_embedding_matrix],
                                     output_dim=char_embedding_matrix.shape[1],
                                     trainable=False)                                                                                                         
    q1_char = Input(shape=(MAX_CHAR_SEQUENCE_LENGTH,), dtype="int32")
    q1_char_embed = SpatialDropout1D(0.3)(char_embedding_layer(q1_char))

    q2_char = Input(shape=(MAX_CHAR_SEQUENCE_LENGTH,), dtype="int32")
    q2_char_embed = SpatialDropout1D(0.3)(char_embedding_layer(q2_char))
    
    char_bilstm = Bidirectional(CuDNNLSTM(200, return_sequences=True))
    
    q1_char_encoded = Dropout(0.3)(char_bilstm(q1_char_embed))
    q1_char_encoded = Concatenate(axis=-1)([q1_char_encoded, q1_char_embed]) 
    q1_char_encoded = GlobalAveragePooling1D()(q1_char_encoded)
    
    q2_char_encoded = Dropout(0.3)(char_bilstm(q2_char_embed))
    q2_char_encoded = Concatenate(axis=-1)([q2_char_encoded, q2_char_embed])
    q2_char_encoded = GlobalAveragePooling1D()(q2_char_encoded)
    
    char_diff  = Lambda(lambda x: K.abs(x[0] - x[1]))([q1_char_encoded, q2_char_encoded]) 
    char_angle = Lambda(lambda x: x[0] * x[1])([q1_char_encoded, q2_char_encoded])   
    
    features_input = Input(shape=[features_train.shape[1]], dtype='float')
    features_dense = BatchNormalization()(features_input)      
    features_dense = Dropout(0.3)(Dense(1024, activation='relu')(features_dense))                                 
    features_dense = BatchNormalization()(features_dense)                                                        
    features_dense = Dense(512, activation='relu')(features_dense) 
    
    # Classifier 
    merged = concatenate([char_diff, char_angle, features_dense])
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization()(merged) 
    merged = Dense(200, activation="relu")(merged)
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization()(merged)
    out    = Dense(1, activation="sigmoid")(merged)
    
    model = Model(inputs=[ q1_char, q2_char, features_input], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='nadam')
    return model

def lstm_cross_char_add_feature_finetune():
    char_embedding_layer = Embedding(name="char_embedding",
                                     input_dim=char_embedding_matrix.shape[0],
                                     weights=[char_embedding_matrix],
                                     output_dim=char_embedding_matrix.shape[1],
                                     trainable=True)
    q1_char = Input(shape=(MAX_CHAR_SEQUENCE_LENGTH,), dtype="int32")
    q1_char_embed = SpatialDropout1D(0.3)(char_embedding_layer(q1_char))

    q2_char = Input(shape=(MAX_CHAR_SEQUENCE_LENGTH,), dtype="int32")
    q2_char_embed = SpatialDropout1D(0.3)(char_embedding_layer(q2_char))
    
    char_bilstm = Bidirectional(CuDNNLSTM(200, return_sequences=True,trainable=False),trainable=False)
    
    q1_char_encoded = Dropout(0.3)(char_bilstm(q1_char_embed))
    q1_char_encoded = Concatenate(axis=-1)([q1_char_encoded, q1_char_embed]) 
    q1_char_encoded = GlobalAveragePooling1D()(q1_char_encoded)
    
    q2_char_encoded = Dropout(0.3)(char_bilstm(q2_char_embed))
    q2_char_encoded = Concatenate(axis=-1)([q2_char_encoded, q2_char_embed])
    q2_char_encoded = GlobalAveragePooling1D()(q2_char_encoded)
    
    char_diff  = Lambda(lambda x: K.abs(x[0] - x[1]))([q1_char_encoded, q2_char_encoded]) 
    char_angle = Lambda(lambda x: x[0] * x[1])([q1_char_encoded, q2_char_encoded])   
    
    features_input = Input(shape=[features_train.shape[1]], dtype='float')
    features_dense = BatchNormalization(trainable=False)(features_input)
    features_dense = Dropout(0.3)(Dense(1024, activation='relu',trainable=False)(features_dense))
    features_dense = BatchNormalization(trainable=False)(features_dense)
    features_dense = Dense(512, activation='relu',trainable=False)(features_dense)
    
    # Classifier 
    merged = concatenate([char_diff, char_angle, features_dense])
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization(trainable=False)(merged)
    merged = Dense(200, activation="relu",trainable=False)(merged)
    
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization(trainable=False)(merged)
    out    = Dense(1, activation="sigmoid",trainable=False)(merged)
    
    model = Model(inputs=[ q1_char, q2_char, features_input], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=0.001))
    return model

model_count = 0    

class RocAucEvaluation(Callback):                          
    def __init__(self, validation_data=(), interval=1):                                                       
        super(Callback, self).__init__()                                                                       
        self.interval = interval                                                                               
        self.X_val, self.y_val = validation_data                                                               
                                                                                                         
    def on_epoch_end(self, epoch, logs={}):                                                                    
        if epoch % self.interval == 0:                                                                         
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=2 ** 9)                              
                                                                                                    
            y_pred = np.array(y_pred)                                                                          
            y_pred = y_pred                                                                                    
                                                                                                    
            real = np.array(self.y_val) 
            score = roc_auc_score(real, y_pred) 
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score)) 

sk_train = np.zeros((train_q1_word_seq.shape[0], 1))

remix_index = []
for train_r_idx, val_idx in StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(y_r, y_r):
    sample_rate = 1.0
    train_r_idx = random.sample(train_r_idx,int(sample_rate * len(train_r_idx)))
    remix_index.append(train_r_idx)
print("add " + str(int(sample_rate * len(train_r_idx))) + " each epoch.")

for train_idx, val_idx in StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(y, y):
    val_q1_word_seq, val_q2_word_seq  = train_q1_word_seq[val_idx], train_q2_word_seq[val_idx]                     
    train_q1_word_seq_, train_q2_word_seq_ = train_q1_word_seq[train_idx], train_q2_word_seq[train_idx]            
    val_q1_char_seq, val_q2_char_seq  = train_q1_char_seq[val_idx], train_q2_char_seq[val_idx]                     
    train_q1_char_seq_, train_q2_char_seq_ = train_q1_char_seq[train_idx], train_q2_char_seq[train_idx]  
    features_train_, features_val = features_train[train_idx], features_train[val_idx]
    train_y, val_y = y[train_idx], y[val_idx]                                                                    

    train_q1_word_seq_2 = np.concatenate([train_q1_word_seq_, train_r_q1_word_seq[remix_index[model_count]]], axis=0)
    train_q2_word_seq_2 = np.concatenate([train_q2_word_seq_, train_r_q2_word_seq[remix_index[model_count]]], axis=0)
    train_q1_char_seq_2 = np.concatenate([train_q1_char_seq_, train_r_q1_char_seq[remix_index[model_count]]], axis=0)
    train_q2_char_seq_2 = np.concatenate([train_q2_char_seq_, train_r_q2_char_seq[remix_index[model_count]]], axis=0)
    features_train_2 = np.concatenate([features_train_, features_train_r[remix_index[model_count]]], axis=0)
    train_y2 = np.concatenate([train_y, y_r[remix_index[model_count]]], axis=0)
    sample_weight = np.concatenate([np.array([1.0] * len(train_y)), np.array([0.20] * len(y_r[remix_index[model_count]]))], axis=0)

    RocAuc = RocAucEvaluation(validation_data=([val_q1_char_seq, val_q2_char_seq, features_val], val_y), interval=1)
    model = lstm_cross_char_add_feature()  
    if model_count == 0: print (model.summary())                                       
    early_stopping = EarlyStopping(monitor="val_loss", patience=8)
    plateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='min', factor=0.5, patience=3)
    best_model_path = "../models/" + "lstm_cross_char_add_feature_augmentation_best_model" + str(model_count) + ".h5"
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
    
    if not os.path.exists(best_model_path):
        hist = model.fit([train_q1_char_seq_2, train_q2_char_seq_2, features_train_2],
                              train_y2,
                              validation_data=([val_q1_char_seq, val_q2_char_seq, features_val], val_y),
                              epochs=60,
                              batch_size=128,
                              shuffle=True,
                              callbacks=[early_stopping, model_checkpoint, RocAuc, plateau],
                              sample_weight=sample_weight,
                              verbose=1)
    
    model = lstm_cross_char_add_feature_finetune()  
    model.load_weights(best_model_path) 
    print model.evaluate([val_q1_char_seq, val_q2_char_seq, features_val], val_y, batch_size=128, verbose=1)
    
    best_model_path = "../models/" + "lstm_cross_char_add_feature_augmentation_best_model_finetune" + str(model_count) + ".h5"
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
    if not os.path.exists(best_model_path):
        hist = model.fit([train_q1_char_seq_2, train_q2_char_seq_2, features_train_2],
                         train_y2,
                         validation_data=([val_q1_char_seq, val_q2_char_seq, features_val], val_y),
                         epochs=60,
                         batch_size=128,
                         shuffle=True,
                         callbacks=[early_stopping, model_checkpoint, RocAuc, plateau],
                         sample_weight=sample_weight,
                         verbose=1)

    model.load_weights(best_model_path)
    print model.evaluate([val_q1_char_seq, val_q2_char_seq, features_val], val_y, batch_size=128, verbose=1)

    sk_train[val_idx] = model.predict([val_q1_char_seq, val_q2_char_seq, features_val])
    print ( roc_auc_score(val_y, sk_train[val_idx]) )
    
    test['y_pre_1_%s' %model_count] = model.predict([test_q1_char_seq, test_q2_char_seq, features_test], batch_size=2 ** 9)
    test[['y_pre_1_%s' %model_count]].rename(columns={'y_pre_%s' %model_count: "y_pre"}).to_csv("../output/lstm_cross_char_add_feature_augmentation_finetune_%s.csv" %model_count)
    test_r_res = model.predict([test_r_q1_char_seq,test_r_q2_char_seq, features_test_r], batch_size=2 ** 9)
    test_r['y_pre_%s' % model_count] = test_r_res.clip(0.000001, 0.999999)
    m = test_r.groupby('oldindex')['y_pre_%s' %model_count].mean()
    first = test_r.groupby('oldindex')['y_pre_%s' %model_count].first()
    test['y_pre_2_%s' % model_count] = ((m * 0.2+first * 0.8)).values
    print test_r['y_pre_%s' % model_count][:10]
    model_count += 1

pred_cols = [col for col in test.columns if 'y_pre_1' in col]
submit = test[pred_cols] 
submit['y_pre'] = submit[pred_cols].mean(axis=1)  
submit.to_pickle('../output/lstm_cross_char_add_feature_augmentation_finetune_10cv_mean.pickle')
submit[['y_pre']].to_csv('../output/lstm_cross_char_add_feature_augmentation_finetune_cv10_mean.csv', index=False, float_format='%.10f')

pred_cols = [col for col in test.columns if 'y_pre_2' in col]
submit = test[pred_cols]
submit['y_pre'] = submit[pred_cols].mean(axis=1)
submit.to_pickle('../output/lstm_cross_char_add_feature_augmentation_finetune_10cv_mean.pickle')
submit[['y_pre']].to_csv('../output/lstm_cross_char_add_feature_augmentation_finetune_cv10_mean_r.csv', index=False, float_format='%.10f')

