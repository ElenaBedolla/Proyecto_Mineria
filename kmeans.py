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
from mpl_toolkits.mplot3d import Axes3D

def continue_finding_elbow(SSE_d2x, i, tol):
    # Condicion 1:
    if( abs(SSE_d2x[i]) >= abs(SSE_d2x[i-1]) ):
        return True
    # Condicion 2:
    elif( abs(SSE_d2x[i]-SSE_d2x[i-1]) < tol ):
        return True
    return False

def plot_clusters(X,labels,colors):        
    for i in range( len(labels) ):
        plt.plot( X[i][0], X[i][1], 'ro', color=colors[labels[i]] )
        
        
def plot_silhouette(X,clustering):
    labels = np.unique( clustering.fit_predict(X) )
    num_clusters = labels.shape[0]
    
    silhouette_vals = silhouette_samples(X, clustering.fit_predict(X), metric='euclidean')

    y_ax_lower, y_ax_upper = 0,0
    yticks = []
    for i,c in enumerate(labels):
        c_silhouette_vals = silhouette_vals[ clustering.fit_predict(X) == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / num_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper),
                 c_silhouette_vals,
                 height=1.0,
                 edgecolor='none',
                 color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    return silhouette_avg

def how_many_clusters(X, max_clusters,colors):
    num_clusters = 0
    SSE          = []
    SSE_d2x      = [] #Esta lista empieza con n_clusters=1
    silhouettes  = []
    
        
    while( num_clusters < max_clusters+1): #Para calcular la derivada del siguiente
        num_clusters += 1
        
        clustering = KMeans(n_clusters=num_clusters, init="k-means++", n_init=10, max_iter=300, random_state=42).fit(X)
        
        if num_clusters < max_clusters+1:
            plt.subplot(max_clusters,2,2*num_clusters-1)
            plot_clusters(X,clustering.labels_,colors)
        
            if( num_clusters > 1 ):
                plt.subplot(max_clusters,2,2*num_clusters)
                silhouettes.append(plot_silhouette(X,clustering))
        
        SSE.append( clustering.inertia_ )
        
        #Calculamos la segunda derivada del punto anterior normalizando ademas para que no influyan las proporciones
        if( num_clusters > 2 ):
            SSE_d2x.append( (SSE[-3]+SSE[-1]-2*SSE[-2]) / (SSE[-3]-SSE[-1]) )
    
    index_silhouettes = np.argsort(-np.array(silhouettes))

    best_d2x=[SSE_d2x[i] for i in index_silhouettes]
    best_d2x.insert(0,0)

    i=1
    while continue_finding_elbow(best_d2x, i, 0.2):
        i+=1
    
    elbow = index_silhouettes[i-1]+1        
        
    return SSE,SSE_d2x, elbow


def apply_KMeans(data_file, data_ohe_file, n):
    # Debido a la gran demanda de memoria y procesamiento por estos calculos, se puede proveer el parametro n para solo usar unca cantidad arbitraria de datos
    
    data = pd.read_csv(data_file).loc[:n]
    text = data['Text_lemmatized']
    vectorizer = CountVectorizer(min_df = 1, lowercase=False)
    X = vectorizer.fit_transform(text)
    X_coo = coo_matrix(vectorizer.fit_transform(text), dtype=np.float64).tocsc()
    
    u, s, vt = svds(X_coo)
    
    
    SVD = u[:,:8] * s[:8]
    
    data_OHE = pd.read_csv(data_ohe_file).loc[:n]
    
    new_X = np.concatenate([SVD, data_OHE.values], axis=1)
    u, s, vt = svd(new_X)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    SVD = u[:,:3] * s[:3]

    ax.plot(SVD[:,0], SVD[:,1], SVD[:,2], 'o')
    st.write(fig)
    
    colors = [[np.random.random(), np.random.random(), np.random.random()] for i in range(6)]
    SSE, SSE_d2x, codo = how_many_clusters(SVD, 5, colors)
    
    st.write("Codo optimo encontrado por el algoritmo: %s"%(codo))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    kmeans = KMeans(n_clusters=codo)

    kmeans.fit(SVD)
    colors = ['r', 'g', 'b', 'y']
    for i, punto in enumerate(SVD):
        ax.plot([punto[0]], [punto[1]], [punto[2]], 'o', color=colors[kmeans.labels_[i]])
        
    st.write(fig)
    
