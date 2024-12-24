#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 19:46:41 2021

@author: tonmoydey
"""

import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx
from networkx.algorithms import bipartite


class TwitterStreamSummarizationMonotone:  

    
    def __init__(self, tweet_keyword_matrix):
        """
        class ImageSummarization:
            INPUTS and INSTANCE VARIABLES:
                image_similarity_matrix (becomes instance variable Sim for short): a 2D symmetric np.array of float64s
        """

        self.groundset       = list( range(tweet_keyword_matrix.shape[0]) )
        self.num_images      = tweet_keyword_matrix.shape[0]
        self.Sim             = tweet_keyword_matrix


       


    def value(self, S):
        ''' Coverage term + genre diversity term '''
        sum_we = (self.Sim[S,:]).sum(axis=0)
        sqrt_sum_we = np.sqrt(sum_we)
        return np.sum(sqrt_sum_we)
        # return np.sum(self.Sim[list(set(S)), :])
    



    def marginalval(self, S, T):
        """ Speed-Optimized Marginal value of adding set S to current set T for value function above (EQ. 2) """
        # Fast approach
        return self.value(list(set().union(S, T))) - self.value(T)


    def printS(self, S):
        """ Print the id of the tweets in the set (of indices) S """
        for idx in S:
            print(idx)




################################################################################
###   Some helper functions to load the image similarity matrix ###
################################################################################

def load_tweet_keyword_matrix(tweet_keyword_mat_fname):
    '''
    INPUTS:
    image_mat_fname: a string filename of the image similarity edgelist
    OUTPUTS: 
    Return a 2d numpy array 
    '''
    edgelist = pd.read_csv(tweet_keyword_mat_fname, delimiter=' ')
    src = edgelist["source"].unique()
    trg = edgelist["target"].unique()
    net_nx = nx.from_pandas_edgelist(edgelist, edge_attr=True, create_using=None)
    net_nx = net_nx.to_undirected()
    print(bipartite.is_bipartite(net_nx))
    try:
        net_nx.remove_edges_from(net_nx.selfloop_edges())
    except:
        net_nx.remove_edges_from(nx.selfloop_edges(net_nx)) #Later version of networkx prefers this syntax
    Sim = bipartite.biadjacency_matrix(net_nx, row_order=src, column_order=trg, dtype=None, weight='weight', format='csr')
    return Sim

















