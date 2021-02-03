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

nltk.download()
nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')

def quitar_puntuacion(x):
    return ''.join([i for i in x if i not in string.punctuation])

def quitar_stopwords(x, stopwords):
  extra_stopwords = ['im', 'also', 'would', 'like', 'really']
  return ' '.join([word for word in x.split(' ') if word not in stopwords+extra_stopwords])

lemmatizer = WordNetLemmatizer()

def lemmatize(x):
    return ' '.join([lemmatizer.lemmatize(word) for word in x.split(' ')])

def Preprocessing(data, data_file):
    data['Text_minusculas'] = data['Review Text'].apply(lambda x: x.lower())
    data['Text_no_punct'] = data['Text_minusculas'].apply(lambda x: quitar_puntuacion(x))
    
    data['Text_no_stop'] = data['Text_no_punct'].apply(lambda x: quitar_stopwords(x, stopwords))
    data['Text_lemmatized'] = data['Text_no_stop'].apply(lambda x: lemmatize(x))

    st.write(data.head())
    
    data.to_csv(data_file)
