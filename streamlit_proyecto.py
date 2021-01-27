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

from streamlit_preprocesamiento import *
from kmeans import *
from reglas import *

nltk.download('stopwords')
nltk.download('wordnet')

data_file = 'data.csv'
data_ohe_file = 'data_OHE.csv'

seccion = st.sidebar.selectbox(
        "Sección:",
        ('Exploracion', 'Preprocesamiento', 'KMeans', 'Reglas de Asociacion')
    )

if seccion == 'Exploracion':
    st.header("Navegue primeramente a la seccion de 'Preprocesamiento' antes de proceder con las secciones de 'KMeans' y 'Reglas de Asociacion'")
    
    data = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
    data.dropna(subset=['Review Text'], inplace=True)
    
    data.drop(columns='Unnamed: 0', inplace=True)
    
    data_hist = data[['Age', 'Rating', 'Recommended IND', 'Positive Feedback Count']].copy()
    hist = data_hist.hist()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    data_freq = data[['Division Name']].copy()
    vals = data_freq.value_counts()
    vals.plot(kind='bar', ax=ax)
    
    st.write(fig)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    data_freq = data[['Department Name']].copy()
    vals = data_freq.value_counts()
    vals.plot(kind='bar', ax=ax)
    st.write(fig)

    fig=plt.figure()
    ax = fig.add_subplot()
    data_freq = data[['Class Name']].copy()
    vals = data_freq.value_counts()
    vals.plot(kind='bar', ax=ax)
    st.write(fig)
    
elif seccion == 'Preprocesamiento':
    data = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
    data.dropna(subset=['Review Text'], inplace=True)
    
    data.drop(columns='Unnamed: 0', inplace=True)
    data_OHE = pd.get_dummies(data, columns=['Division Name', 'Department Name', 'Class Name']).drop(columns=['Clothing ID', 'Title', 'Review Text'])
    data_OHE.to_csv(data_ohe_file)
    Preprocessing(data, data_file)
    
elif seccion == 'KMeans':
    apply_KMeans(data_file, data_ohe_file, 10000)
    
elif seccion == 'Reglas de Asociacion':
    rules_df, sets_df = calc_rules(data_file)
    
    st.write('Gracias a A priori y las reglas de asociacion, es posible ver el cambio de frecuencia de una palabra dependiendo de la calificacion')
    
    #rating = int(st.radio('Calificacion', ('1', '2', '3', '4', '5')))
    
    #fig = rule_heatmap(rules_df, sets_df, rating)
    #st.write(fig)
    
    #word = st.text_input('Palabra a buscar...', 'love')
    supports_df = plot_word_freq(sets_df, 'love')
    fig=plt.figure()
    ax = fig.add_subplot()
    supports_df.plot.bar(ax=ax)
    st.write(fig)
    
