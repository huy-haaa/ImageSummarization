#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:07:43 2021

@author: tonmoydey
"""

import pandas as pd 
import numpy as np
import networkx as nx

experiment_string = 'CAROADFULL'
        # Weighted Directed highway adjacency matrix
        #filename = "Pems_Adj_thresh_15mi_1044n.csv"
        #filename = "data/caltrans/Pems_Adj_thresh_10mi_522n.csv"
filename = "../data/Pems_Adj.csv" #"../data/Pems_Adj.csv"# This is the FULL NETWORK it is large -- ~1800 nodes

A = pd.read_csv(filename).values

S = [0, 1, 2, 7,10]

a = A[S]
b = A[:,S]
c = A[S][:,S]
G = nx.from_numpy_matrix(A)

nx.write_weighted_edgelist(G, "../data/Pems_Adj_edgelist.txt", delimiter=' ')
# X = np.sum(A[S]) + np.sum(A[:,S]) - np.sum(A[S][:,S])