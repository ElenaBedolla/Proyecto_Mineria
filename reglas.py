import streamlit as st
import pandas as pd
from pathlib import Path

import numpy as np
import string
from math import *
from collections import *

import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
from scipy.linalg import svd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from matplotlib import cm

import matplotlib.pyplot as plt

import seaborn as sns

def obtain_rules(data, min_support, max_len=None):
    transactions = []
    for i, row in data.iterrows():
        transactions.append(row.Text_lemmatized.split(' '))
    te = TransactionEncoder()

    ohe_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(ohe_array, columns=te.columns_)

    frequent_sets = apriori(df, min_support=min_support, max_len=max_len, use_colnames=True)
    interesting_rules = association_rules(frequent_sets, metric = 'lift', min_threshold = 1)
    #interesting_rules.sort_values(by='lift', ascending=False)
    frequent_sets['size'] = frequent_sets['itemsets'].apply(lambda x: len(x))
    interesting_rules['ant_size'] = interesting_rules['antecedents'].apply(lambda x: len(x))
    interesting_rules['con_size'] = interesting_rules['consequents'].apply(lambda x: len(x))

    return frequent_sets, interesting_rules

def plot_word_freq(sets_df, word):
    supports = []
    for set_df in sets_df:
        found=False
        for i in range(set_df.values.shape[0]):
            if word in set_df.values[i][1] and set_df.values[i][2] == 1:
                support = set_df.values[i][0]
                supports.append(support)
                found=True
        if found == False:
            supports.append(0)

    supports_df = pd.DataFrame({'support' : supports})
    #print(supports_df)
    #supports_df.plot(kind='bar', ax=ax)
    return supports_df

#@st.cache
def rule_heatmap(rules_df, sets_df, rating):
    df = rules_df[rating-1].copy()
    set_df = sets_df[rating-1].copy()
    set_df['word'] = set_df['itemsets'].apply(lambda x: list(x)[0])
    set_df.set_index('word', inplace=True)
    df['antecedents_aux'] = df['antecedents'].apply(lambda x: list(x)[0])
    df['consequents_aux'] = df['consequents'].apply(lambda x: list(x)[0])
    unique_vals = pd.unique(df['antecedents_aux']) 
    for i in range(len(unique_vals)):
        for j in range(len(unique_vals)):
            i_val = unique_vals[i]
            j_val = unique_vals[j]
            # Por alguna razon esta parte del codigo provoca algun error, por lo cual la diagonal del mapa de calor se mostrara negra
            #if i_val == j_val:
            #    row = {}
            #    for column in df.columns:
            #        if column == 'antecedents_aux':
            #            row[column] = [i_val]
            #        elif column == 'consequents_aux':
            #            row[column] = [j_val]
            #        elif column == 'support':
            #            row[column] = [np.float64(set_df.loc[i_val]['support'])]
                        #if len(row[column]) > 1:
                        #    print('hola')
            #        else:
            #            row[column] = [np.NaN]
            #    df = pd.concat([df, pd.DataFrame(row)], axis=0)
            if df.loc[(df['antecedents_aux'] == i_val) & (df['consequents_aux'] == j_val)].shape[0] == 0:
                row = {}
                for column in df.columns:
                    if column == 'antecedents_aux':
                        row[column] = [i_val]
                    elif column == 'consequents_aux':
                        row[column] = [j_val]
                    elif column == 'support':
                        row[column] = [np.float64(0.0)]
                    else:
                        row[column] = [np.NaN]
                df = pd.concat([df, pd.DataFrame(row)], axis=0)
    #df.reset_index(inplace=True)
    #print(df[['support', 'antecedents_aux', 'consequents_aux']])
    figure, ax = plt.subplots(figsize=(17, 20))
    ax = sns.heatmap(df.pivot("antecedents_aux", "consequents_aux", "support"))
    return figure

def calc_rules(data_file):
    data = pd.read_csv(data_file)
    
    rules_df = []
    sets_df = []
    for i in range(1,6):
        frequent_sets, interesting_rules = obtain_rules(data.loc[data['Rating'] == i].copy(), min_support=0.03, max_len=2)
        rules_df.append(interesting_rules)
        sets_df.append(frequent_sets)
        
    return rules_df, sets_df
    
    
