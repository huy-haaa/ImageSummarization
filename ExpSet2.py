#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime
import pandas as pd
from scipy import sparse
import networkx as nx
import random
import sys
#import ast
import os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

from mpi4py import MPI

# # Load the Objective function classes 
from objectives import  NetCover,\
                        NetCoverSparse,\
                        TrafficCoverDirWeighted,\
                        RevenueMaxOnNet,\
                        MovieRecommenderMonotoneCover,\
                        InfluenceMaxSparse,\
                        GraphGenerators, \
                        ImageSummarizationMonotone, \
                        TwitterStreamSummarizationMonotone
"""
Provide the full system path to the "SubmodularData/submodular/src" (FAST source code) directory
"""
# FAST_src_path = "/home/tonmoy/SubmodularData/submodular/src"
FAST_src_path = ""

# Load our optimization algorithms and helper functions
from src import submodular_PGB
if (FAST_src_path!=""):
    sys.path.append(FAST_src_path)
    import submodular

def run_PGB(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, p_root=0, trials=2, gph = True):
    procs_vec = [size]*len(k_vals_vec)
    size_groundset = len([ele for ele in objective.groundset])
    n_vec = [size_groundset]*len(k_vals_vec)
    comm.barrier();
    algostring = 'PGB'
    OPT = None
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    eps = 0.1
    val_vec = []
    queries_vec = []
    time_vec = []

    for ii, kk in enumerate(k_vals_vec):

        for trial in range(trials):
            comm.barrier()

            # Free memory between trials
            sol = []
            sol_r = [] 
            time_r = []
            queries_r = []

            # Run the algorithm
            val, queries, time, sol, sol_r, time_r, queries_r = submodular_PGB.ParallelGreedyBoost(objective, kk, eps, comm, rank, size, p_root, seed=trial, stop_if_approx=False );
            time_PGB = time;
            val_PGB = val;
            
            if rank == p_root:
            
                print('f(S)=', val, 'queries=', queries, 'time=', time, algostring, experiment_string, 'with k=', kk)

                val_vec.append(val)
                queries_vec.append(queries)
                time_vec.append(time)

                ## Save data progressively
                dataset = pd.DataFrame({'f_of_S':  val_vec, \
                                        'Queries': queries_vec, \
                                        'Time':    time_vec, \
                                        'k':       np.concatenate([np.repeat(k_vals_vec[:ii], trials), [kk]*(trial+1)]), \
                                        'n':       [size_groundset]*(ii*trials+trial+1), \
                                        'nproc':   [size]*(ii*trials+trial+1), \
                                        'trial':   np.concatenate([np.tile(range(1,(trials+1)), ii), range(1, (trial+2))])
                                        })
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp2_'+ algostring +'_'+ str(size)+".csv", index=False)    

    if rank == p_root:
        print('Completed ', algostring, 'for experiment: ', experiment_string)


def run_FAST(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, p_root=0, trials=2, gph = True):
    comm.barrier()
    procs_vec = [size]*len(k_vals_vec)
    size_groundset = len([ele for ele in objective.groundset])
    n_vec = [size_groundset]*len(k_vals_vec)
    comm.barrier()
    algostring = 'FAST'
    if rank == p_root:
        print('Beginning ', algostring, 'for experiment: ', experiment_string)
    # eps = 0.007
    eps = 0.025
    val_vec = []
    queries_vec = []
    time_vec = []
    # Save data progressively. 
    for ii, kk in enumerate(k_vals_vec):

        for trial in range(trials):
            comm.barrier()

            # Free memory between trials
            sol = []
            sol_r = [] 
            time_r = []
            queries_r = []

            # Run the algorithm
            val, queries, time, sol = submodular.FAST_guessopt_parallel(objective, kk, eps, comm, rank, size, \
                                            preprocess_add=True, lazy_binsearch=True, lazyouterX=True, 
                                            debug_asserts=False, weight_sampling_eps=1.0, sample_threshold=True, \
                                            lazyskipbinsearch=True, verbose=False, stop_if_approx=True, \
                                            eps_guarantee=0.1, p_root=0, seed=trial)

            time_FAST = time;
            val_FAST = val;
            if rank == p_root:
                val_vec.append(val)
                queries_vec.append(queries)
                time_vec.append(time)

                ## Save data progressively
                dataset = pd.DataFrame({'f_of_S':  val_vec, \
                                        'Queries': queries_vec, \
                                        'Time':    time_vec, \
                                        'k':       np.concatenate([np.repeat(k_vals_vec[:ii], trials), [kk]*(trial+1)]), \
                                        'n':       [size_groundset]*(ii*trials+trial+1), \
                                        'nproc':   [size]*(ii*trials+trial+1), \
                                        'trial':   np.concatenate([np.tile(range(1,(trials+1)), ii), range(1, (trial+2))])
                                        })
                dataset.to_csv(path_or_buf = filepath_string + experiment_string +'_exp2_'+ algostring +'_'+ str(size)+".csv", index=False)


            if rank == p_root:
                print('f(S)=', val, 'queries=', queries, 'time=', time, algostring, experiment_string, 'k=', kk)
                print('\n')
    if rank == p_root:
        print('Completed ', algostring, 'for experiment: ', experiment_string) 

def run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo = "ALL", p_root=0, trials=2):
    """ Parallel MPI function to run all benchmark algorithms over all values of k for a given objective function and 
    save CSV files of data and runtimes """
    if(algo=="ALL"):
        run_PGB(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)

        if (FAST_src_path!=""):
            run_FAST(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)
        else:
            print("FAST code system path not provided")

        

    if(algo=="PGB"):
        run_PGB(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)
     
    
    if(algo=="FAST"):
        if (FAST_src_path!=""):
            run_FAST(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)
        else:
            print("FAST code system path not provided")    
    
    #run_PGB(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)
        
    #run_FAST(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)   
    
    if rank == p_root:
        print('FINISHED\n')
    comm.barrier()

    return


if __name__ == '__main__':

    start_runtime = datetime.now()

    p_root = 0

    filepath_string = "experiment_results_output_data/"
    
    obj = str( sys.argv[1] );
    algoIn = str( sys.argv[2] );

    # Start MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == p_root:
        print('Initializing run')


    # ##################################################################
    # ##################################################################
    # ######                 SYNTHETIC GRAPH DATA                 ######
    # ##################################################################
    # ##################################################################
    size_of_ground_set = 100000

    k_vals_vec = [1000]
    
    
    # # ###################################################################################
    # # ## Boolean Undirected VertCover ER Example PARALLEL GENERATED #####################
    # # ###################################################################################
    if(obj=="ER"):
        experiment_string = 'ER_100k'
        comm.barrier()
        if rank == p_root:
            print( 'Initializing ER Objective' )
    
    
        # Set the seed before generating random ER matrix to synchronize all processors' random draws
        np.random.seed(42)
        
    
        # Set p == prob of an edge in Erdos Renyi
        p = 0.0001
    
        #Generate the ER Adj. Matrix. Use sparse matrix if A is big (matrix operations slower in sparse format)
        A = GraphGenerators.ErdosRenyiSymBoolSparseCSRparallel(size_of_ground_set, p, comm, rank, size)
        #A = GraphGenerators.ErdosRenyiSymBoolSparseCSR(size_of_ground_set, p)
        A.setdiag(1)
    
        # Generate our NetCover class containing the function
        objective = NetCoverSparse.NetCoverSparse(A)
    
        ## For testing:
        if rank == p_root:
            print( 'ER Objective initialized. Beginning tests.' )
    
    
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo=algoIn)
    
        # A = A.todense()
        # objective = NetCover.NetCover(A)
        # run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)
    
        comm.barrier()
        print('\n')



    # #################################################################
    # ## Boolean Undirected Watts-Strogatz Example ####################
    # #################################################################
    if(obj=="WS"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing WS Objective' )
        experiment_string = 'WS_100k'
        k_vals_vec = [1000]
        if rank == p_root:
    
            G = nx.watts_strogatz_graph(n=size_of_ground_set, k=10, p=0.1, seed=42)
            try:
                G.remove_edges_from(G.selfloop_edges())
            except:
                G.remove_edges_from(nx.selfloop_edges(G)) #Later version of networkx prefers this syntax
    
    
            if size_of_ground_set < 100:
                A = np.asarray( nx.to_numpy_matrix(G) )
                A.fill_diagonal(1)
                A.astype('bool_')
                # print( 'size of A', A.shape[0])
                # print( 'density of A', np.sum(A)/len(A)**2)
                # Generate our NetCover class containing the function
                objective_rootprocessor = NetCover.NetCover(A)
    
            else:
                A = nx.to_scipy_sparse_matrix(G, format='csr')
                A.setdiag(1)
                # Generate our NetCover class containing the function
                objective_rootprocessor = NetCoverSparse.NetCoverSparse(A)
            
    
        # Send class to all processors
        if rank != 0:
            objective_rootprocessor = None
        objective = comm.bcast(objective_rootprocessor, root=0)
    
        if rank == p_root:
            print( 'Watts-Strogatz Objective initialized. Beginning tests.' )
        comm.barrier()
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo=algoIn)




    # ################################################################
    # ## Boolean Undirected VertCover BA Example #####################
    # ################################################################
    if(obj=="BA"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing BA Objective' )
        experiment_string = 'BA_100k'
        k_vals_vec = [1000]
    
        if rank == p_root:
    
            G = nx.barabasi_albert_graph(size_of_ground_set, 5, seed=42)
            try:
                G.remove_edges_from(G.selfloop_edges())
            except:
                G.remove_edges_from(nx.selfloop_edges(G)) #Later version of networkx prefers this syntax
    
            if size_of_ground_set < 100:
                A = np.asarray( nx.to_numpy_matrix(G) )
                A.fill_diagonal(1)
                A.astype('bool_')
                objective_rootprocessor = NetCover.NetCover(A)
    
            else:
                A = nx.to_scipy_sparse_matrix(G, format='csr')
                A.setdiag(1)
                # Generate our NetCover class containing the function
                objective_rootprocessor = NetCoverSparse.NetCoverSparse(A)
            
    
        # Send class to all processors
        if rank != 0:
            objective_rootprocessor = None
        objective = comm.bcast(objective_rootprocessor, root=0)
    
        if rank == p_root:
            print( 'BA Objective initialized. Beginning tests.' )
        comm.barrier()
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo=algoIn)
    
        comm.barrier()
    
    
    # # ###################################################################################
    # # ## Boolean Undirected VertCover SBM Example PARALLEL GENERATED ####################
    # # ###################################################################################
    if(obj=="SBM"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing SBM Objective' )
        experiment_string = 'SBM_48k'
        k_vals_vec = [1000]
        
        #Generate the SBM Adj. Matrix
        comm.barrier()
        #objective_rootprocessor = None
        np.random.seed(42) #4 gives 100k nodes; 42 gives 48k; good plot did 48k
        
        # k_vals_vec = [50, 500, 1000, 2500, 5000, 7500, 10000]#, 1500, 2000]
        numberC = 50 # Number of clusters
        minSize = 100# min cluster size
        maxSize = 5000 # max cluster size
        p = 0.1  # = prob of edge # was 0.0001
        
        if rank == p_root:
            # Generate this large graph in parallel!
            A = GraphGenerators.SBMSparseCSRparallel(numberC, minSize, maxSize, p, comm, rank, size)
            A.setdiag(1)
            # Generate our NetCover class containing the function
            objective_rootprocessor = NetCoverSparse.NetCoverSparse(A)
           
        # Send class to all processors
        if rank != 0:
            objective_rootprocessor = None
        objective = comm.bcast(objective_rootprocessor, root=0)
        ## For testing:
        if rank == p_root:
            print( 'SBM Objective initialized. Beginning tests.' )
            print ('size of ground set:', len(objective.groundset))
        
        comm.barrier()
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo=algoIn)




    # # # ##################################################################
    # # # ##################################################################
    # # # ###############           REAL DATA            ###################
    # # # ##################################################################
    # # # ##################################################################

    
    # ##############################################################
    # ## DIRECTED EdgeCover ON CALI ROAD NETWORK EXPERIMENT ###########
    # ##############################################################
    if(obj=="TRF"):
        experiment_string = 'CAROADFULL'
        # Weighted Directed highway adjacency matrix
        #filename = "Pems_Adj_thresh_15mi_1044n.csv"
        #filename = "data/caltrans/Pems_Adj_thresh_10mi_522n.csv"
        filename = "data/Pems_Adj.csv"# This is the FULL NETWORK it is large -- ~1800 nodes
    
        A = pd.read_csv(filename).values
    
        # Generate our DIRECTED MaxCut class containing the function
        objective = TrafficCoverDirWeighted.TrafficCoverDirWeighted(A)
    
        k_vals_vec = [500] # gets up to 72% Pems_Adj.csv"
    
        # Print info and start the stopwatch
        if rank == p_root:
            print ('Starting California experiment. Network size = ', str(A.shape[0]), ' nodes.')
        comm.barrier()
    
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo=algoIn)
        
        comm.barrier()
    



    # ################################################################
    # ##           INfluencemax NETWORK Example EPINIONS        ##############
    # ################################################################
    if(obj=="IFM"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing influence max Objective' )
    
        experiment_string = 'INFLUENCEEPINIONS'
    
        # Undirected Facebook Network. Format as an adjacency matrix
        filename_net = "data/soc-epinions.csv"
    
    
        edgelist = pd.read_csv(filename_net)
        net_nx = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=None)
        net_nx = net_nx.to_undirected()
        try:
            net_nx.remove_edges_from(net_nx.selfloop_edges())
        except:
            net_nx.remove_edges_from(nx.selfloop_edges(net_nx)) #Later version of networkx prefers this syntax
    
    
        #A = np.asarray( nx.adjacency_matrix(net_nx).todense() )
        if rank == p_root:
            print( 'Loaded data. Generating sparse adjacency matrix' )
        A = nx.to_scipy_sparse_matrix(net_nx, format='csr')
        A.setdiag(1)
        #A = A[0:1000,0:1000] # restrict to mini version for testing
    
        #objective = NetCover.NetCover(A)
        p = 0.01
        objective = InfluenceMaxSparse.InfluenceMaxSparse(A, p)
        if rank == p_root:
            print( 'EPINIONS Objective of', A.shape[0], 'elements initialized. Beginning tests.' )
    
        comm.barrier()
    
        k_vals_vec = [5000]
    
        comm.barrier()
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo=algoIn)





    # ################################################################
    # ## YOUTUBE 2000 REVENUE MAXIMIZATION EXAMPLE ####################
    # ################################################################
    if(obj=="RVM"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing Youtube Objective' )
        experiment_string = 'YOUTUBE2000'
    
        # Top 500 communities have ~15k nodes
        edgelist = pd.read_csv('data/youtube_2000rand_edgelist.csv', delimiter=',')
        #edgelist = pd.read_csv('data/youtube/youtube_top50_edgelist.csv', delimiter=',')
        # Edgelist to Adjacency Matrix
        A = edgelist.pivot(index = "source", columns = "target", values = "weight_draw")
        # Cast to Numpy Matrix
        A = A.values
        # Missing edges are 0
        A[np.isnan(A)] = 0
        A[A>0] = A[A>0] + 1.0
    
        # Set the power between 0 and 1. More towards 0 means revenue is more subadditive as a node gets more influences
        alpha = 0.3
    
        # Generate class containing our f(S)
        objective = RevenueMaxOnNet.RevenueMaxOnNet(A, alpha)
        if rank == p_root:
            print( 'YOUTUBE Objective initialized. Adjacency matrix shape is:', A.shape, ' Beginning tests.' )
    
    
        k_vals_vec = [1000]
    
        comm.barrier()
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)




    # ################################################################
    # ###           Movie Recommendation Example  WITH COVER      ##############
    # ################################################################
    # Load the image data and generate the image pairwise distance matrix Dist
    if(obj=="MVR"):
        comm.barrier()
        experiment_string = 'MOVIEBIGCOVER'
    
        num_random_movies_users = -1 #-1 means 'all images'
    
        # Initialize the movie data
        movie_user_mat_fname = "data/Movie_ratings_mat.csv"
        movies_dat_fname = 'data/movies.dat'
    
    
        objective = None
    
        try:
            Sim = MovieRecommenderMonotoneCover.load_movie_user_matrix(movie_user_mat_fname)
            if rank != p_root:
                Sim = None # Free memory on all but root proc -- we just try to have them load to test whether the movie data exists
    
            if rank == p_root:
                genres_idx, genres_dict, genres_strings, movie_titles, movie_years = \
                                    MovieRecommenderMonotoneCover.load_movie_genres(movies_dat_fname)
    
                movierandstate = np.random.RandomState(42)
    
                if num_random_movies_users > 0:  
                    movie_rows     = movierandstate.choice(Sim.shape[0], num_random_movies_users, replace=False)
                    Sim            = Sim[movie_rows,:][:,movie_rows]
                    genres_idx     = genres_idx[movie_rows]
                    genres_strings = genres_strings[movie_rows]
                    movie_titles   = movie_titles[movie_rows]
    
                print('loaded movie ratings matrix of shape', Sim.shape)
    
                # Generate our class containing the function
                genre_weight = 0.5 * np.max( np.sum(Sim, 1) ) # adding a new genre equally valuable to 1/100 the average sum of a movie's ratings
                year_weight = 0.0 * np.max( np.sum(Sim, 1) ) # adding a new debut year equally valuable to 1/100 the average sum of a movie's ratings
                ratings_weight = 1.0
                cover_weight = 1.0
                good_movie_score = 4.5
    
                objective = MovieRecommenderMonotoneCover.MovieRecommenderMonotoneCover(Sim, movie_titles, genres_idx, \
                                                                                movie_years, ratings_weight, cover_weight, \
                                                                                genre_weight, year_weight, good_movie_score)
    
            # Root processor broadcasts the objective function to all processors
            objective = comm.bcast(objective, p_root)
            
            k_vals_vec = [1000]
    
            comm.barrier()
    
            run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo=algoIn)
    
            comm.barrier()
    
        except:
            if rank == p_root:
                print('\n\nThe final experiment (movie recommendation) requires a large movies data file (too large for GitHub).\
                        Add this file to the /data/ folder to run this experiment.\n\n')
    
    # ################################################################
    # ###           Image Summarisation Example  WITH COVER      ##############
    # ################################################################
    # Load the image data and generate the image pairwise distance matrix Dist
    
    if(obj=="IS_PGB"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing image summ Objective' )
    
        experiment_string = 'IMAGESUMM'
    
        # Undirected Facebook Network. Format as an adjacency matrix
        filename_net = "data/images_10K_mat.csv"
        
        Sim = ImageSummarizationMonotone.load_image_similarity_matrix(filename_net)
        
        
        if rank == p_root:
            print( 'Loaded data. Generated adjacency matrix' )
        objective = ImageSummarizationMonotone.ImageSummarizationMonotone(Sim)
        
        if rank == p_root:
            print( ' Objective of', Sim.shape[0], 'images initialized. Beginning tests.' )
    
        comm.barrier()
    
        k_vals_vec = [1000]
    
        comm.barrier()
        #run_PGB(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo="PGB")
    
    
    if(obj=="IS_FAST"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing image summ Objective' )
    
        experiment_string = 'IMAGESUMM'
    
        # Undirected Facebook Network. Format as an adjacency matrix
        filename_net = "data/images_10K_mat.csv"
        
        Sim = ImageSummarizationMonotone.load_image_similarity_matrix(filename_net)
        
        
        if rank == p_root:
            print( 'Loaded data. Generated adjacency matrix' )
        objective = ImageSummarizationMonotone.ImageSummarizationMonotone(Sim)
        
        if rank == p_root:
            print( ' Objective of', Sim.shape[0], 'images initialized. Beginning tests.' )
    
        comm.barrier()
    
        k_vals_vec = [1000]
    
        comm.barrier()
        #run_FAST(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)   
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo="FAST")
    
    # ################################################################
    # ###           Twitter Summarisation Example       ##############
    # ################################################################
    # Load the image data and generate the image pairwise distance matrix Dist
    
    ##           INfluencemax NETWORK Example EPINIONS        ##############
    # ################################################################
    if(obj=="TS_PGB"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing twitter summ Objective' )
    
        experiment_string = 'TWITTERSUMM'
    
        # Undirected Facebook Network. Format as an adjacency matrix
        filename_net = "data/tweets_keyword_edgelist.txt"
        
        Sim = TwitterStreamSummarizationMonotone.load_tweet_keyword_matrix(filename_net)
        
        
        if rank == p_root:
            print( 'Loaded data. Generating sparse adjacency matrix' )
        objective = TwitterStreamSummarizationMonotone.TwitterStreamSummarizationMonotone(Sim)
        
        if rank == p_root:
            print( ' Objective of', Sim.shape[0], 'tweets initialized. Beginning tests.' )
    
        comm.barrier()
    
        k_vals_vec = [1000]
    
        comm.barrier()
        #run_PGB(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo="PGB")
    
    
    if(obj=="TS_FAST"):
        comm.barrier()
        if rank == p_root:
            print( 'Initializing twitter summ Objective' )
    
        experiment_string = 'TWITTERSUMM'
    
        # Undirected Facebook Network. Format as an adjacency matrix
        filename_net = "data/tweets_keyword_edgelist.txt"
        
        Sim = TwitterStreamSummarizationMonotone.load_tweet_keyword_matrix(filename_net)
        
        
        if rank == p_root:
            print( 'Loaded data. Generating sparse adjacency matrix' )
        objective = TwitterStreamSummarizationMonotone.TwitterStreamSummarizationMonotone(Sim)
        
        if rank == p_root:
            print( ' Objective of', Sim.shape[0], 'tweets initialized. Beginning tests.' )
    
        comm.barrier()
    
        k_vals_vec = [1000]
    
        comm.barrier()
        #run_FAST(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size)
        
        run_LTLG_experiments(objective, k_vals_vec, filepath_string, experiment_string, comm, rank, size, algo="FAST")
        comm.barrier()
        
    if rank == p_root:
        print ('\n\nALL EXPERIMENTS COMPLETE, total minutes elapsed =', (datetime.now()-start_runtime).total_seconds()/60.0,'\n\n')

