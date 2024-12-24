#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''


from datetime import datetime
import numpy as np
import random
from queue import PriorityQueue

try:
    from mpi4py import MPI
except ImportError:
    MPI = None 
if not MPI:
    print("MPI not loaded from the mpi4py package. Serial implementations will function, \
            but parallel implementations will not function.")

def check_inputs(objective, k):
    '''
    Function to run basic tests on the inputs of one of our optimization functions:
    '''
    # objective class contains the ground set and also value, marginalval methods
    assert( hasattr(objective, 'groundset') )
    assert( hasattr(objective, 'value') )
    assert( hasattr(objective, 'marginalval') )
    # k is greater than 0
    assert( k>0 )
    # k is smaller than the number of elements in the ground set
    assert( k<len(objective.groundset) )
    # the ground set contains all integers from 0 to the max integer in the set
    assert( np.array_equal(objective.groundset, list(range(np.max(objective.groundset)+1) )) )


def sample_seq( X, k, randstate ):
    if len(X) <= k:
        randstate.shuffle(X)
        return X
    Y = list(randstate.choice(X, k, replace=False));
    randstate.shuffle(Y);
    return Y;

def parallel_margvals_returnvals_fls(objective, L, N, comm, rank, size):
    '''
    Parallel-compute the marginal value f(element)-f(L) of each element in set N. Version for stochasticgreedy_parallel.
    Returns the ordered list of marginal values corresponding to the list of remaining elements N
    '''
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    comm.barrier()
    N_split_local = np.array_split(N, size)[rank]

    # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    valL = objective.value( L );
    ele_vals_local_vals = [ objective.value( list( set().union([elem], L ) ) ) - valL for elem in N_split_local ]
    
    # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_vals_local_vals)
    return [val for sublist in ele_vals for val in sublist]




def parallel_margvals_returnvals(objective, L, N, comm, rank, size):
    '''
    Parallel-compute the marginal value f(element)-f(L) of each element in set N. Version for stochasticgreedy_parallel.
    Returns the ordered list of marginal values corresponding to the list of remaining elements N
    '''
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    comm.barrier()
    N_split_local = np.array_split(N, size)[rank]

    # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    ele_vals_local_vals = [ objective.marginalval( [elem], L ) for elem in N_split_local ]
    
    # Gather the partial results to all processes
    ele_vals = comm.allgather(ele_vals_local_vals)
    return [val for sublist in ele_vals for val in sublist]




def make_lmbda(eps, k, n):
    '''
    Generate the vector lambda based on LINE 9 in ALG. 1
    '''
    idx = 1;
    lmbda = [ idx ];
    while (idx < n - 1):
        if ( idx < k ):
            newidx = np.floor(idx*(1 + eps));
            if (newidx == idx):
                newidx = idx + 1;
            if (newidx > k):
                newidx = k;
        else:
            newidx = np.floor(idx + eps * k);
            if (newidx == idx):
                newidx = idx + 1;
            if (newidx > n - 1):
                newidx = n - 1;

        lmbda.append( int(newidx) );
        idx = newidx;
    return lmbda




def parallel_adaptiveAdd(lmbda, V, S, objective, eps, k, comm, rank, size, tau = 0):
    
    '''
    Parallel-compute the marginal value of block  of elements and set the corresponding flag to True based on condition in LINE 13 of ALG. 1
    '''
    
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    comm.barrier()
    idcs = np.array_split( range( len( lmbda ) ), size )[ rank ]
    #lmbda_split_local = np.array_split(lmbda, size)[rank]

    B = []
    if ( len( idcs ) > 0 ):
        pos=lmbda[ idcs[ 0 ] ];
        ppos=0
        if pos > 1:
            ppos=lmbda[ idcs[0] - 1 ];

        oldtmpS = list( set().union( V[0 : ppos], S) );
        valTmpS = objective.value( oldtmpS );
        Ti=list(set(V[ppos:pos]))

        for idx in range(1,len(idcs) + 1):
            tmpS = list( set(oldtmpS) | set( Ti) );
            
            # gain= objective.value( tmpS ) - valTmpS;
            gain= objective.marginalval( tmpS, oldtmpS)
            if (tau == 0):
                thresh = len(Ti)*(1-eps)*valTmpS / np.float(k);
            else:
                thresh = len(Ti)*(1-eps)*tau;
                
            if (gain >= thresh):
                B.append(True)
            else:
                B.append(False)

            valTmpS = valTmpS + gain;
            if (idx >= len(idcs)):
                posEnd = lmbda[ idcs[ -1 ] ];
            else:
                posEnd = lmbda[ idcs[ idx ] ];
            Ti=V[ lmbda[ idcs[ idx - 1] ]: posEnd ];

    
    # Gather the partial results to all processes
    B_vals = comm.allgather(B)

    return [val for sublist in B_vals for val in sublist]





def parallel_threshold_sample(V, S, objective, tau, eps, delta, k, comm, rank, size, randstate, pastGains):
    
    '''
    The parallelizable greedy algorithm THRESHOLDSEQ for Fixed Threshold 'tau' (Algorithm 3)
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    list V - contains the elements currently in the groundset
    list S - contains the elements currently in the solution set
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    float tau -- the fixed threshold
    float eps -- the error tolerance between 0 and 1
    float delta -- the probability tolerance between 0 and 1
    int k -- the cardinality constraint (must be k>0)
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    randstate -- random seed to use when drawing samples 
    pastGains -- each element is the current marginal gain of the corresponding element in groundset

    OUTPUTS:
    list pastGains -- each element is the current marginal gain of the corresponding element in groundset
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    list S -- the solution, where each element in the list is an element with marginal values > tau.

    '''  
    
    comm.barrier()

    ell = int(np.ceil( (4.0 + 8.0/ eps ) * np.log( len(V) / delta ) ) ) + 1;
    V = list( np.sort( list( set(V)-set(S) ) ) );
    queries = 0
    for itr in range( ell ):
        s = np.min( [k - len(S), len(V) ] );
        
        seq = sample_seq( V, s, randstate );

        lmbda = make_lmbda( eps, s, s );
        
        B = parallel_adaptiveAdd(lmbda, seq, S, objective, eps, k, comm, rank, size, tau)
        #Added query increment
        queries += len(lmbda)
        lmbda_star = lmbda[0]
        if len(B) > 1:
            for i in range(1,len(B)):
                if(B[i]):
                    if (i == len(B) - 1):
                        lmbda_star = lmbda[-1];
                else:
                    lmbda_star = lmbda[i]
                    break;

        #Add elements
        #T= parallel_pessimistically_add_x_seq(objective, S, seq, tau, comm, rank , size );
        T = set(seq[0:lmbda_star]);
        
        for i in range(lmbda_star, len(B)):
            if (B[i]):
                T = set().union(T, seq[ lmbda[ i - 1 ] : lmbda[ i ] ]);
        S= list(set().union(S, T))
        V = list( np.sort( list( set(V)-set(S) ) ) );
        #if (rank == 0):
            #print( "Lambda_Star: " , lmbda_star, len(S) );
            #print( "starting filter..." );
        #Filter
        gains = parallel_margvals_returnvals(objective, S, V, comm, rank, size)
        #Added query increment
        queries += len(V)
        #if (rank == 0):
#            print("done.");
        for ps in range( len(gains )):
            pastGains[ V[ps] ] = gains[ ps ];
        
        V_above_thresh = np.where(gains >= tau)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        if (len(V) == 0):
            break;
        if (len(S) == k):
            break;

    if (len(V) > 0 and len(S) < k):
        if (rank == 0):
            print( "ThresholdSample has failed. This should be an extremely rare event. Terminating program..." );
            print( "V: ", V );
            print( "S: ", S );
        exit(1);
            
    return [pastGains, S, queries];




def LinearSeq(objective, k, eps, comm, rank, size, p_root=0, seed=42, stop_if_approx=True):
    '''
    The preprocessing algorithm LINEARSEQ for Submodular Mazimization (Algorithm 1)
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_approx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution

    OUTPUTS:
    float f(S) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list S -- the solution, where each element in the list is an element in the solution set.
    list of lists S_rounds -- each element is a list containing the solution set S at the corresponding round.
    list of lists time_rounds -- each element is a list containing the time at the corresponding round
    list of lists query_rounds -- each element is a list containing the number of queries at the corresponding round
    list singletonVals -- each element is the current marginal gain of the corresponding element in groundset
    '''    
    comm.barrier()
    p_start = MPI.Wtime()
    n = len(objective.groundset)

    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    queries = 0
    time0 = datetime.now()

    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    iters = 0
    #I1 = make_I(eps, k)

    currGains = parallel_margvals_returnvals(objective, [], [ele for ele in objective.groundset], comm, rank, size)
    #Added query increment
    queries += len(objective.groundset)
    #currGains = parallel_margvals_returnvals(objective, [], [ele for ele in objective.groundset], comm, rank, size)

    #initialize S to top singleton
    S = [np.argmax(currGains)];
    singletonIdcs = np.argsort(currGains);
    singletonVals = currGains;
    currGains = np.sort(currGains);
    #run first considering the top singletons as the universe
    V = np.array(objective.groundset)[ singletonIdcs[-3*k:] ]

    currGains = currGains[-3*k:];
    valtopk = np.sum( currGains[-k:] );

    while len(V) > 0:
        # Filtering
        t = objective.value(S)/np.float(k)
        # Added increment
        queries += 1
        if stop_if_approx:
            if (rank == 0):
                print( "checking ratio: ", 0.5*valtopk, t * k );
            if (t >= 0.5 * valtopk / np.float(k)):
                if (rank == 0):
                    print( "FLS stopping early, approx reached." );
                if(len(S)>k):
                    Ap = S[len(S) - k : len(S)]
                else:
                    Ap = S;
                valAp = objective.value(Ap)
                queries += 1
                #Tracking Adaptivity
                S_rounds.append([ele for ele in S])
                query_rounds.append(queries)  
                comm.barrier()
                p_stop = MPI.Wtime()
                time = (p_stop - p_start)
                if rank == p_root:
                    print ('FLS:', valAp, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(Ap))
    
                return valAp, queries, time, S, S_rounds, time_rounds, query_rounds, singletonVals
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        #now recompute requisite gains
        if (rank == 0):
            print( len(V) );
            print( "starting pmr..." );
        currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        #Added query increment
        queries += len(V)
        #currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        if ( rank == 0):
            print( "done.");
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];
        ###End Filtering
        
        if (rank == p_root):
            print( len(V) );

        if (len(V) == 0):
            break;
        
        # Radnom Permutation
        randstate.shuffle(V);
        lmbda = make_lmbda( eps, k, len(V) );

        if (rank == p_root):
            print("starting adaptiveAdd...");
        B = parallel_adaptiveAdd(lmbda, V, S, objective, eps, k, comm, rank, size)
        
        if (rank == p_root):
            print("done.");
        #Added query increment
        queries += len( lmbda );
        lmbda_star = lmbda[0]
        
        if len(B) > 1:
            n_GB = 1
            unionT = 1
            for i in range(1,len(B)):
                if(B[i]):
                    n_GB += 1
                    unionT += lmbda[i] - lmbda[i-1]
                    if (i == len(B) - 1):
                        lmbda_star = lmbda[-1];
                else:
                    if(lmbda[i - 1]<=k):
                        if(n_GB == i):
                            lmbda_star = lmbda[i - 1]
                    else:
                        if(unionT >= k):
                            lmbda_star = lmbda[i - 1]
                    n_GB = 0
                    unionT = 0


        T = set(V[0:lmbda_star])
        for i in range(lmbda_star, len(B)):
            if (B[i]):
                T = set().union(T, V[ lmbda[ i - 1 ] : lmbda[ i ] ]);
        if (rank == p_root):
            print("done.");
        S= list(set().union(S, T))
        #Tracking adaptivity
        S_rounds.append([ele for ele in S])
        query_rounds.append(queries)
        time_rounds.append( MPI.Wtime() - p_start ) 

        if (rank == p_root):
            print( "Lambda_Star: " , len(T) );

        
        
    t = objective.value(S) / np.float( k );
    queries += 1
    # if(lazy):
    V_above_thresh = np.where(singletonVals >= t)[0]
    V = [ objective.groundset[idx] for idx in V_above_thresh ]; 
    currGains = [ singletonVals[idx] for idx in V_above_thresh ];
    
    while len(V) > 0:
        # Filtering
        t = objective.value(S)/np.float(k)
        queries += 1
        if stop_if_approx:
            if (rank == 0):
                print( "checking ratio: ", 0.5*valtopk, t * k );
            if (t >= 0.5 * valtopk / np.float(k)):
                if (rank == 0):
                    print( "FLS stopping early, approx reached." );
                if(len(S)>k):
                    Ap = S[len(S) - k : len(S)]
                else:
                    Ap = S;
                valAp = objective.value(Ap)
                queries += 1
                comm.barrier()
                p_stop = MPI.Wtime()
                time = (p_stop - p_start)
                if rank == p_root:
                    print ('FLS:', valAp, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(Ap))
    
                return valAp, queries, time, S, S_rounds, time_rounds, query_rounds, singletonVals
       
        # if(lazy):
        
        #do lazy discard first, based on last round's computations
        V_above_thresh = np.where(currGains >= t)[0]
        V = list( set([ V[idx] for idx in V_above_thresh ] ));

        #now recompute requisite gains
        if (rank == 0):
            print( len(V) );
            print( "starting pmr..." );
        currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        #Added query increment
        queries += len(V)
        #currGains = parallel_margvals_returnvals(objective, S, [ele for ele in V], comm, rank, size)
        if ( rank == 0):
            print( "done.");
        V_above_thresh = np.where(currGains >= t)[0]
        V = [ V[idx] for idx in V_above_thresh ];
        currGains = [ currGains[idx] for idx in V_above_thresh ];

        if (len(V) == 0):
            break;
        
        if (rank == p_root):
            print( len(V) );
        # Radnom Permutation
        randstate.shuffle(V);
        lmbda = make_lmbda( eps, k, len(V) );

        if (rank == p_root):
            print("starting adaptiveAdd...");
        B = parallel_adaptiveAdd(lmbda, V, S, objective, eps, k, comm, rank, size)
        
        if (rank == p_root):
            print("done.");
        
        queries += len( lmbda );
        lmbda_star = lmbda[0]
        
        if len(B) > 1:
            n_GB = 1
            unionT = 1
            for i in range(1,len(B)):
                if(B[i]):
                    n_GB += 1
                    unionT += lmbda[i] - lmbda[i-1]
                    if (i == len(B) - 1):
                        lmbda_star = lmbda[-1];
                else:
                    if(lmbda[i - 1]<=k):
                        if(n_GB == i):
                            lmbda_star = lmbda[i - 1]
                    else:
                        if(unionT >= k):
                            lmbda_star = lmbda[i - 1]
                    n_GB = 0
                    unionT = 0


        T = list(set(V[0:lmbda_star]))
        #T = parallel_pessimistically_add_x_seqVar( objective, S, V,k, comm, rank, size );
        if (rank == p_root):
            print("done.");
        S= list(set().union(S, T))
        #Tracking adaptivity
        S_rounds.append([ele for ele in S])
        query_rounds.append(queries)
        time_rounds.append( MPI.Wtime() - p_start ) 

        if (rank == p_root):
            print( "Lambda_Star: " , len(T) );

        
    
    if(len(S)>k):
        Ap = S[len(S) - k : len(S)]
    else:
        Ap = S;
    valAp = objective.value(Ap)
    queries += 1

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    if rank == p_root:
        print ('FLS:', valAp, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(Ap))
    
    return valAp, queries, time, S, S_rounds, time_rounds, query_rounds, singletonVals





def ParallelGreedyBoost(objective, k, eps, comm, rank, size, p_root=0, seed=42,stop_if_approx=True):

    '''
    The parallelizable greedy algorithm PARALLELGREEDYBOOST to Boost to the Optimal Ratio. (Algorithm 4)
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    bool stop_if_approx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution

    OUTPUTS:
    float f(S) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    ''' 
    
    comm.barrier()
    p_start = MPI.Wtime()
    eps_FLS = 0.21
    Gamma, queries, time, sol, sol_r, time_r, queries_r, pastGains = LinearSeq(objective, k, eps_FLS, comm, rank, size, p_root, seed, True );
    
    #alpha = 1.0 / (4.0 + (2*(3+(2*eps_FLS)))/(1 - (2*eps_FLS) - (2*(eps_FLS ** 2)))*eps_FLS);
    alpha = 1.0 / (4.0 + (4*(2-eps_FLS))/((1 - eps_FLS)*(1-(2*eps_FLS))) *eps_FLS);
    valtopk = np.sum( np.sort(pastGains)[-k:] );
    stopRatio = (1.0 - 1.0/np.exp(1) - eps)*valtopk;
    #stopRatio = 0.75*valtopk;
    #stopRatio = 0.85*valtopk;
    if stop_if_approx:
        if Gamma >= stopRatio:
            comm.barrier()
            p_stop = MPI.Wtime()
            time = (p_stop - p_start)
            valSol = Gamma
            if (rank == 0):
                print("ABR stopping early.");

            return Gamma, queries, time, sol, sol_r, time_r, queries_r
    
    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    randstate = np.random.RandomState(seed)
    time0 = datetime.now()

    S = []
    #Tracking adaptivity
    S_rounds = sol_r
    time_rounds = [0]
    query_rounds = queries_r
    iters = 0
    #I1 = make_I(eps, k)
    
    tau = Gamma / (alpha * np.float(k));
    taumin = Gamma / (3.0 * np.float(k));
    V = [ ele for ele in objective.groundset ];

    if (tau > valtopk / np.float(k)):
        tau = valtopk / np.float(k);
    
    #pastGains = np.inf*np.ones(len(V));
    while (tau > taumin):
        tau = np.min( [tau, np.max( pastGains ) * (1.0 - eps)] );
        if (tau <= 0.0):
            tau = taumin;
        V_above_thresh = np.where(pastGains >= tau)[0]
        V_above_thresh = [ V[ele] for ele in V_above_thresh ];
        V_above_thresh = list( set(V_above_thresh) - set(S) );
        if (rank == 0):
            print( tau, taumin, len(S), len( V_above_thresh ) );
        
        currGains = parallel_margvals_returnvals(objective, S, V_above_thresh, comm, rank, size)
        #Added query increment
        queries += len(V_above_thresh)

#        if (rank == 0):
#            print( currGains );
#            print( V_above_thresh );
        for ps in range( len(V_above_thresh )):
            pastGains[ V_above_thresh[ps]] = currGains[ ps ];
        V_above_thresh_id = np.where(currGains >= tau)[0]
        V_above_thresh = [ V_above_thresh[ele] for ele in V_above_thresh_id ];
        if (rank == p_root):
            print( len(V_above_thresh) );
        if (len(V_above_thresh) > 0):
            [pastGains, S, queries_tmp] = parallel_threshold_sample( V_above_thresh, S, objective, tau, eps / np.float(3), 1.0 / ((np.log( alpha / 3 ) / np.log( 1 - eps ) ) + 1) , k, comm, rank, size, randstate, pastGains );
            #Added query increment
            queries += queries_tmp
            #Tracking adaptivity
            S_rounds.append([ele for ele in S])
            query_rounds.append(queries)
            for ele in S:
                pastGains[ele] = 0;

        if (len(S) >= k):
            break;
        if stop_if_approx:
            if objective.value(S) >= stopRatio:
                comm.barrier()
                p_stop = MPI.Wtime()
                time = (p_stop - p_start)
                valSol = objective.value( S )
                #Added increment
                queries += 1
                query_rounds[-1]=queries
                if (rank == 0):
                    print("ABR stopping early.");
                return valSol, queries, time, sol, sol_r, time_r, queries_r
    
    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    valSol = objective.value( S )
    # Added increment
    queries += 1
    if rank == p_root:
        print ('ABR:', valSol, queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    
    return valSol, queries, time, S, S_rounds, time_rounds, query_rounds




def lazygreedy_parallel(objective, k, comm, rank, size):
    ''' 
    Accelerated lazy greedy algorithm: for k steps (Minoux 1978).
    **NOTE** solution sets and values may be different than those found by our Greedy implementation, 
    as the two implementations break ties differently (i.e. when two elements 
    have the same marginal value,  the two implementations may not pick the same element to add)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    '''
    k_max = np.max(k)
    check_inputs(objective, k_max)
    queries = 0
    time0 = datetime.now()
    L = []
    N = [ele for ele in objective.groundset]
    #k_max = np.max(k)
    val_vec = []
    time_vec = []
    qry_vec = []
    # On the first iteration, LazyGreedy behaves exactly like regular Greedy:
    ele_vals = parallel_margvals_returnvals(objective, [], objective.groundset, comm, rank, size)
    queries += len(N)
    valL = 0;
    cue = PriorityQueue();
    maxGain = np.max( ele_vals );
    for idx in range(len(ele_vals)):
        cue.put( ( maxGain - ele_vals[ idx ], idx ) );

    # On remaining iterations, we update values lazily
    for i in range(0,k_max+1):
        if (rank == 0):
            print('LazyGreedy round', i, 'of', k)

        (oldgain, ele) = cue.get();
        oldgain = maxGain - oldgain;
        
        gainEle = objective.value( list( set().union( L, [ele] ) ) ) - valL;
        queries += 1
        while gainEle < oldgain:
            cue.put( ( maxGain - gainEle, ele ) );

            (oldgain, ele) = cue.get();
            oldgain = maxGain - oldgain;
        
            gainEle = objective.value( list( set().union( L, [ele] ) ) ) - valL;
            queries += 1
            
        L = list( set().union(L, [ele]) );
        valL = valL + gainEle;
        if i in k:
           print("Adding ",i," to vecs") 
           val_vec.append(valL)
           time_i = (datetime.now() - time0).total_seconds()
           time_vec.append(time_i)
           qry_vec.append(queries)
           
    return val_vec, qry_vec, time_vec