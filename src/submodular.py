'''
@author: Adam Breuer
'''
from datetime import datetime
import numpy as np

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




def topk(objective, k):
    ''' 
    Returns the solution containing the top k elements with the highest marginal contribution to the empty set.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint

    OUTPUTS:
    float f(L) -- the avg. value of the solution
    int queries -- the total queries (== numsamples)
    float time -- the processing time to optimize the function.
    list S -- the solution set
    '''
    check_inputs(objective, k)
    time0 = datetime.now()
    marginalvals =   [ objective.value([ele]) for ele in objective.groundset ]
    queries = len(objective.groundset)
    S = np.array(objective.groundset)[ np.argsort(marginalvals)[-k:] ]
    time = (datetime.now() - time0).total_seconds()
    return objective.value(S), queries, time, S




def topk_parallel(objective, k, comm, rank, size):
    ''' 
    Returns the solution containing the top k elements with the highest marginal contribution to the empty set.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    float f(L) -- the avg. value of the solution
    int queries -- the total queries (== numsamples)
    float time -- the processing time to optimize the function.
    list S -- the solution set
    '''
    check_inputs(objective, k)
    comm.barrier()
    p_start = MPI.Wtime()
    marginalvals = parallel_margvals_returnvals(objective, [], objective.groundset, comm, rank, size)

    queries = len(objective.groundset)
    S = np.array(objective.groundset)[ np.argsort(marginalvals)[-k:] ]
    comm.barrier()
    time = MPI.Wtime() - p_start
    return objective.value(S), queries, time, S




def randomksolutionval(objective, k, numsamples):
    ''' 
    Returns the value of a random set of size k
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    int numsamples -- the number of random solutions of size k to draw. We return the mean of this.
    
    OUTPUTS:
    float f(L) -- the avg. value of the solution
    int queries -- the total queries (== numsamples)
    float time -- the processing time to optimize the function.
    '''
    check_inputs(objective, k)
    time0 = datetime.now()
    f_k = ( np.mean( [ objective.value(sampleX(objective.groundset,k)) for i in range(numsamples) ] ))
    queries = numsamples
    time = (datetime.now() - time0).total_seconds()
    return f_k, queries, time




def randomksolutionval_parallel(objective, k, numsamples, comm, rank, size, seed=42):
    ''' 
    PARALLEL: Returns the value of a random set of size k
       
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    int numsamples -- the number of random solutions of size k to draw. We return the mean of this.
    
    OUTPUTS:
    float f(L) -- the avg. value of the solution
    int queries -- the total queries (== numsamples)
    float time -- the processing time to optimize the function.
    '''
    check_inputs(objective, k)
    # Parallel time
    comm.barrier()
    p_start = MPI.Wtime()

    # Each processor gets UNIQUE local random seed to draw unique samples
    random_state_local_unq = np.random.RandomState(rank+seed)

    # Each processor does numsamples/size samples
    numsamples_local = int(np.ceil(numsamples/np.float(size)))
    rand_sets = [random_state_local_unq.choice(objective.groundset, k, replace=False) for ss in range(numsamples_local)]
    f_k_local = ( np.mean( [ objective.value(rand_sets[s]) for s in range(numsamples_local) ] ))

    # Reduce to global mean
    f_k = (1.0/size)*comm.allreduce(f_k_local, op=MPI.SUM)

    queries = size*numsamples_local
    return f_k, queries, MPI.Wtime() - p_start




def lazygreedy(objective, k):
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
    check_inputs(objective, k)
    queries = 0
    time0 = datetime.now()
    L = []
    L_rounds = [[]] # to hold L's evolution over rounds. Sometimes the new round does not add an element. 
    queries_vec = []
    k_vals_vec_prog = []
    time_rounds = [0]
    query_rounds = [0]
    N = [ele for ele in objective.groundset]

    # On the first iteration, LazyGreedy behaves exactly like regular Greedy:
    ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
    ele_vals_sortidx = np.argsort(ele_vals)[::-1]
    bestVal_idx = ele_vals_sortidx[0]
    L.append( N[bestVal_idx] )
    queries += len(N)
    lazy_idx = 1

    # On remaining iterations, we update values lazily
    for i in range(1,k):
        if i%25==0:
            print('LazyGreedy round', i, 'of', k)

        # If new marginal value of best remaining ele after prev iter exceeds the *prev* marg value of 2nd best, add it
        queries += 1
        if objective.marginalval( [ N[ ele_vals_sortidx[ lazy_idx ] ] ], L ) >= ele_vals[ ele_vals_sortidx[1+lazy_idx] ]:
            #print('Found a lazy update')
            L.append( N[ele_vals_sortidx[lazy_idx]] )
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx += 1

        else:
            # If we did any lazy update iterations, we need to update bookkeeping for ground set
            N = list(set(N) - set(L)) 

            # Compute the marginal addition for each elem in N, then add the best one to solution L; 
            # Then remove it from remaning elements N
            ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
            ele_vals_sortidx = np.argsort(ele_vals)[::-1]
            bestVal_idx = ele_vals_sortidx[0]
            L.append( N[bestVal_idx] )
            #print('LG adding this value to solution:', ele_vals[bestVal_idx] )

            queries += len(N)
            L_rounds.append([ele for ele in L])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            lazy_idx = 1

    time = (datetime.now() - time0).total_seconds()
    return objective.value(L), queries, time, L, L_rounds, time_rounds, query_rounds




def lazierthanlazygreedy(objective, k, eps, randseed=42):
    ''' 
    Accelerated lazy stochastic greedy algorithm: for k steps (Mirzasoleiman et al. 2015).
    (i.e. Stochastic Greedy with lazy updates)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    
    OPTIONAL INPUTS:
    randseed -- a seed to seed the random set generator. Setting this causes the function to replicate the same result.
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    list of lists L_rounds -- each element is a list containing the solution set L at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in L_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    '''
    check_inputs(objective, k)

    queries = 0
    time0 = datetime.now()
    n = len(objective.groundset)
    randstate = np.random.RandomState(randseed)

    L = []
    L_rounds = [[]] # to hold L's evolution over rounds.
    time_rounds = [0]
    query_rounds = [0]
    s = int( np.ceil(n/np.float(k) * np.log(1.0/eps)) )
    assert(s>0)

    lazy_idx = 1
    lazyvals = np.inf*np.ones(len(objective.groundset))

    N = [ele for ele in objective.groundset]

    for i in range(k):
        # Draw a random set of remaining elements to evaluate this round
        if len(N) > s:
            R = randstate.choice(N, s, replace=False)
        else:
            R = np.copy(N)

        # Lazy add to S if possible
        # Get last known (lazy) marginal value of each element in R
        R_lazyvals = lazyvals[R]
        R_lazyvals_sortidx = np.argsort(R_lazyvals)
        R_lazyvals_best_ele = R[R_lazyvals_sortidx[-1]]

        queries += 1
        # If new marginal value of best element exceeds previous (lazy) marginal value of other ele's in R, add it
        if (len(R_lazyvals_sortidx) >= 2) and \
                (objective.marginalval( [R_lazyvals_best_ele], L ) >= R_lazyvals[R_lazyvals_sortidx[-2]]):
            #print('Lazy')
            L.append( R_lazyvals_best_ele )
            N.remove( R_lazyvals_best_ele )

        # Compute the marginal value for each element in R, then add the best one to solution L; 
        # Then remove it from remaning elements N
        else:
            ele_vals = [ objective.marginalval( [elem], L ) for elem in R ]
            queries += len(R)
            bestVal_idx = np.argmax(ele_vals)
            L.append( R[bestVal_idx] )
            N.remove( R[bestVal_idx] )

            # Update lazy values
            lazyvals[R] = ele_vals

        L_rounds.append([ele for ele in L])
        time_rounds.append((datetime.now() - time0).total_seconds())
        query_rounds.append(queries)


        if i%25==0:
            print('Lazier than lazy greedy round', i, 'of', k)

    time = (datetime.now() - time0).total_seconds()
    return objective.value(L), queries, time, L, L_rounds, time_rounds, query_rounds





def lazierthanlazygreedynaive_parallel(objective, k, eps, comm, rank, size, p_root=0, randseed=42):
    ''' 
    PARALLELIZED Accelerated lazy stochastic greedy algorithm: for k steps (Mirzasoleiman et al. 2015).
    (i.e. Stochastic Greedy with lazy updates)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    
    OPTIONAL INPUTS:
    int p_root: the int (a valid rank) of the processor we will use as the master/root processor
    randseed -- a seed to seed the random set generator. Setting this causes the function to replicate the same result.
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    list of lists L_rounds -- each element is a list containing the solution set L at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in L_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    '''
    check_inputs(objective, k)

    n = len(objective.groundset)
    queries_vec = []
    time_vec = []
    k_vals_vec_prog = []
    queries = 0
    comm.barrier()
    p_start = MPI.Wtime()

    L_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]

    if rank == p_root:
        randstate = np.random.RandomState(randseed)
    else:
        randstate = None

    lazyvals = np.inf*np.ones(len(objective.groundset))

    # All processors need to know current solution at each iter in order to evaluate marginal val of new elements
    L = []
    N = [ele for ele in objective.groundset]
    assert( eps > 0 ), 'eps must be greater than 0!'
    s = int( np.ceil(n/np.float(k) * np.log(1.0/eps)) )
    if rank == p_root:
        print('lazierthanlazygreedy sample complexity per round: s=', s)

    for i in range(k):
        # Root elem draws the random set of ground set elements to test this round, then broadcasts to other processors
        # (but first, root elem attempts a lazy update)
        queries += 1
        if rank == p_root:
            if len(N) > s:
                R = randstate.choice(N, s, replace=False)
                #fevals_perproc += np.ceil(np.float(len(R))/np.float(size)) ## DELETEME
            else:
                R = np.copy(N)

            R_lazyvals = lazyvals[R]
            R_lazyvals_sortidx = np.argsort(R_lazyvals)
            R_lazyvals_best_ele = R[R_lazyvals_sortidx[-1]]

            if (len(R_lazyvals_sortidx) >= 2) and \
                (objective.marginalval( [R_lazyvals_best_ele], L ) >= R_lazyvals[R_lazyvals_sortidx[-2]]):
                # Lazy update
                #print('Lazy')
                R = [R_lazyvals_best_ele]

        else:
            R = None
        R = comm.bcast(R, root=p_root)

        # Compute the marginal value for each element in R, then add the best one to solution L; remove it from remaining elements N
        if len(R) > 1:
            ele_vals = parallel_margvals_returnvals(objective, L, R, comm, rank, size)
            best_ele = R[np.argmax(ele_vals)]
            L.append( best_ele )
            N.remove( best_ele )
            queries += len(R)

            L_rounds.append( [ele for ele in L] )
            time_rounds.append( MPI.Wtime() - p_start )
            query_rounds.append(queries)

            #Update lazy values
            lazyvals[R] = ele_vals
        else:
            L.append( R[0] )
            N.remove( R[0] )

            L_rounds.append( [ele for ele in L] )
            time_rounds.append( MPI.Wtime() - p_start )
            query_rounds.append(queries)

        if ((rank == p_root) and (i%100==0)):
            print(i, 'of', k)

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)

    # if rank == p_root:
    #     print('fevals_perproc', fevals_perproc) ## DELETEME

    #if rank == p_root:
    return objective.value(L), queries, time, L, L_rounds, time_rounds, query_rounds




def lazierthanlazygreedy_parallel(objective, k, eps, comm, rank, size, p_root=0, randseed=42):
    ''' 
    PARALLELIZED Accelerated lazy stochastic greedy algorithm: for k steps (Mirzasoleiman et al. 2015).
    (i.e. Stochastic Greedy with lazy updates)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    
    OPTIONAL INPUTS:
    int p_root: the int (a valid rank) of the processor we will use as the master/root processor
    randseed -- a seed to seed the random set generator. Setting this causes the function to replicate the same result.
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    list of lists L_rounds -- each element is a list containing the solution set L at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in L_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    '''
    check_inputs(objective, k)

    assert(size>=2), 'This function requires at least 2 processors in parallel. \
                        To run on one, use the alternative lazierthanlazygreedy() function.'

    n = len(objective.groundset)
    queries_vec = []
    time_vec = []
    k_vals_vec_prog = []
    queries = 0
    comm.barrier()
    p_start = MPI.Wtime()

    L_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]


    if rank == p_root:
        randstate = np.random.RandomState(randseed)
    else:
        randstate = None

    lazyvals = np.inf*np.ones(len(objective.groundset))

    # All processors need to know current solution at each iter in order to evaluate marginal val of new elements
    L = []
    N = [ele for ele in objective.groundset]
    assert( eps > 0 ), 'eps must be greater than 0!'
    s = int( np.ceil(n/np.float(k) * np.log(1.0/eps)) )

    if rank == p_root:
        print('lazierthanlazygreedy sample complexity per round: s=', s)

    for i in range(k):
        # Root elem draws the random set of ground set elements to test this round, then broadcasts to other processors
        # (but first, root elem attempts a lazy update)
        R = None
        best_ele = None
        R_lazyvals_best_ele = None
        lazyupdate = False

        if rank == p_root:
            if len(N) > s:
                R = randstate.choice(N, s, replace=False)
                #fevals_perproc += np.ceil(np.float(len(R))/np.float(size)) ## DELETEME
            else:
                R = np.copy(N)
            assert(len(R)>1)

            R_lazyvals = lazyvals[R]
            R_lazyvals_sortidx = np.argsort(R_lazyvals)
            R_lazyvals_best_ele = R[R_lazyvals_sortidx[-1]]

            # Remove this previously-best element from R before broadcasting, as root node will check its value
            R = np.delete(R, np.argwhere(R==R_lazyvals_best_ele) )


        # Root node broadcasts elements to other processors
        R = comm.bcast(R, root=p_root)


        # Root node attempts the lazy update
        queries += 1
        if rank == p_root:
            marg_val_best_ele = objective.marginalval( [R_lazyvals_best_ele], L )

            if (len(R_lazyvals_sortidx) >= 2) and (marg_val_best_ele >= R_lazyvals[R_lazyvals_sortidx[-2]]):
                lazyupdate = True
                #print('lazy')
            margval_rank = None


        else:
            # Other processors also each try 1 marginal value while root checks the lazy update
            if len(R) >= rank: 
                ele_rank = R[rank-1]
                margval_rank = objective.marginalval( [ele_rank], L )
            else:
                margval_rank = None
            


        # All nodes ask root node whether it's a lazy update
        lazyupdate = comm.bcast(lazyupdate, root=p_root)
        if lazyupdate:
            best_ele = comm.bcast(R_lazyvals_best_ele, root=p_root)


        # Otherwise, root node gathers the margvals in R that other processors computed and sends out the rest to be computed
        else:
            queries += len(R)-1
            first_p_margvals = comm.gather(margval_rank, root=p_root)
            # if rank == p_root:
            #     print('first_p_margvals', first_p_margvals, 'len_firstpmargval=', len(first_p_margvals))

            if rank == p_root:
                #print(first_p_margvals)
                first_p_margvals = [val for val in first_p_margvals if val is not None] # drop the None from the root processor

            # All processors then split up the remaining elements that have not yet been checked and compute their marginal values
            if len(R) > size-1:
                ele_vals_fromp = parallel_margvals_returnvals(objective, L, R[(size-1):], comm, rank, size)
            else:
                ele_vals_fromp = [-np.inf]

            if rank == p_root:
                #first_p_margvals = first_p_margvals[1:] 
                # print(marg_val_best_ele)
                # print(first_p_margvals)
                # print(ele_vals_fromp)
                if marg_val_best_ele > np.max(list(set().union(first_p_margvals, ele_vals_fromp))):
                    best_ele = R_lazyvals_best_ele

                elif np.max(first_p_margvals) > np.max(ele_vals_fromp):
                    best_ele = R[ np.argmax(first_p_margvals) ]

                else:
                    best_ele = R[ (np.argmax(ele_vals_fromp)+size-1) ]

        # All processors learn the overall best element in R from the root processor
        #(so they can update current solution and do margvals next round)
        best_ele = comm.bcast(best_ele, root=p_root)

        # ele_vals_global_vals_max, ele_vals_global_max_ele = parallel_margvals_faster(objective, L, R, comm, rank, size)

        L.append( best_ele )
        N.remove( best_ele )

        L_rounds.append( [ele for ele in L] )
        time_rounds.append( MPI.Wtime() - p_start )
        query_rounds.append(queries)

        #Update lazy values
        if rank == p_root:
            lazyvals[ R_lazyvals_best_ele ] = marg_val_best_ele
            lazyvals[ R[:len(first_p_margvals)] ] = first_p_margvals
            lazyvals[ R[len(first_p_margvals):] ] = ele_vals_fromp

        if ((rank == p_root) and (i%100==0)):
            print(i, 'of', k)

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)

    # if rank == p_root:
    #     print('fevals_perproc', fevals_perproc) ## DELETEME

    #if rank == p_root:
    return objective.value(L), queries, time, L, L_rounds, time_rounds, query_rounds



def greedy(objective, k):
    ''' 
    Greedy algorithm: for k steps.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    list of lists L_rounds -- each element is a list containing the solution set L at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in L_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    '''
    check_inputs(objective, k)

    queries = 0
    time0 = datetime.now()

    L = []
    N = [ele for ele in objective.groundset]

    L_rounds = []
    time_rounds = [0]
    query_rounds = [0]

    for i in range(k):
        if i%25==0:
            print('Greedy round', i, 'of', k)
        # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaning elements N
        ele_vals = [ objective.marginalval( [elem], L ) for elem in N ]
        bestVal_idx = np.argmax(ele_vals)
        L.append( N[bestVal_idx] )
        #print('G adding this value to solution:', ele_vals[bestVal_idx] )

        queries += len(N)
        N.remove( N[bestVal_idx] )
        L_rounds.append([ele for ele in L])
        time_rounds.append((datetime.now() - time0).total_seconds())
        query_rounds.append(queries)

    val = objective.value(L)
    time = (datetime.now() - time0).total_seconds()

    return val, queries, time, L, L_rounds, time_rounds, query_rounds




def stochasticgreedy(objective, k, eps, randseed=42):
    ''' 
    Stochastic greedy algorithm: for k steps (Mirzasoleiman et al. 2015).
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    
    OPTIONAL INPUTS:
    randseed -- a seed to seed the random set generator. Setting this causes the function to replicate the same result.
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    '''
    check_inputs(objective, k)

    queries = 0
    time0 = datetime.now()
    n = len(objective.groundset)
    randstate = np.random.RandomState(randseed)

    L = []
    L_rounds = [[]] # to hold L's evolution over rounds. 
    time_rounds = [0]
    query_rounds = [0]
    s = int( np.ceil(n/np.float(k) * np.log(1.0/eps)) )

    N = [ele for ele in objective.groundset]
    #print len(N)   
    for i in range(k):
        # if i%25==0:
        #     print 'Stochastic greedy round', i, 'of', k

        # Draw a random set of remaining elements to evaluate this round
        if len(N) > s:
            R = randstate.choice(N, s, replace=False)
        else:
            R = np.copy(N)

        # Compute the marginal value for each element in R, then add the best one to solution L; remove it from remaning elements N
        ele_vals = [ objective.marginalval( [elem], L ) for elem in R ]
        bestVal_idx = np.argmax(ele_vals)
        L.append( R[bestVal_idx] )
        queries += len(R)
        N.remove( R[bestVal_idx] )

        L_rounds.append([ele for ele in L])
        time_rounds.append( (datetime.now() - time0).total_seconds() )
        query_rounds.append(queries)

        if i%25==0:
            print('Stochastic greedy round', i, 'of', k, 'len(N)', len(N), 'len(R)', len(R), 's', s)

    time = (datetime.now() - time0).total_seconds()
    return objective.value(L), queries, time, L, L_rounds, time_rounds, query_rounds




def stochasticgreedy_parallel(objective, k, eps, comm, rank, size, p_root=0, randseed=42):
    ''' 
    PARALLEL Stochastic greedy algorithm: for k steps (Mirzasoleiman et al. 2015).
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    
    OPTIONAL INPUTS:
    int p_root: the int (a valid rank) of the processor we will use as the master/root processor
    randseed -- a seed to seed the random set generator. Setting this causes the function to replicate the same result.
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    '''
    check_inputs(objective, k)

    n = len(objective.groundset)
    queries_vec = []
    time_rounds = [0]
    query_rounds = [0]
    k_vals_vec_prog = []
    queries = 0
    comm.barrier()
    p_start = MPI.Wtime()

    fevals_perproc = 0 ## DELETEME

    if rank == p_root:
        randstate = np.random.RandomState(randseed)
    else:
        randstate = None

    # All processors need to know current solution at each iter in order to evaluate marginal val of new elements
    L = []
    L_rounds = [[]]
    N = [ele for ele in objective.groundset]
    assert( eps>0 ), 'eps must be greater than 0!'
    s = int( np.ceil(n/np.float(k) * np.log(1.0/eps)) )

    for i in range(k):

        # Root elem draws the random set of ground set elements to test this round, then broadcasts to other processors:
        if rank == p_root:
            if len(N) > s:
                R = randstate.choice(N, s, replace=False)
                fevals_perproc += np.ceil(np.float(len(R))/np.float(size)) ## DELETEME
            else:
                R = np.copy(N)
        else:
            R = None
        R = comm.bcast(R, root=0)

        # Compute the marginal value for each element in R, then add the best one to solution L; remove it from remaining elements N
        # ele_vals_global_vals_max, ele_vals_global_max_ele = parallel_margvals_forSG(objective, L, R, comm, rank, size)

         # Compute the marginal value for each element in R, then add the best one to solution L; remove it from remaining elements N
        if len(R) > 1:
            ele_vals = parallel_margvals_returnvals(objective, L, R, comm, rank, size)
            best_ele = R[np.argmax(ele_vals)]
            L.append( best_ele )
            N.remove( best_ele )
            queries += len(R)

        else:
            L.append( R[0] )
            N.remove( R[0] )

        L_rounds.append([ele for ele in L])
        time_rounds.append(MPI.Wtime()-p_start)
        query_rounds.append(queries)

        # L.append( ele_vals_global_max_ele )
        # N.remove( ele_vals_global_max_ele )
        # queries += len(R)

        if ((rank == p_root) and (i%100==0)):
            print(i, 'of', k)

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)

    if rank == p_root:
        print('fevals_perproc', fevals_perproc) ## DELETEME

    return objective.value(L), queries, time, L, L_rounds, time_rounds, query_rounds




def greedy_parallel(objective, k, comm, rank, size, progsave_data=False, k_vals_vec=[], filepath_string='',\
                     experiment_string='', algostring='', p_root=0):

    ''' 
    Standard Greedy algorithm, but marginal values are computed in parallel each round.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    bool progsave_data -- if True, then the algorithm saves data after each round listed in the list k_vals_vec
    list of ints k_vals_vec -- see above
    string filepath_string -- used to make filenames if using progsave_data
    string experiment string -- used to make filenames if using progsave_data
    string algostring -- used to make filenames if using progsave_data

    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list L -- the solution, where each element in the list is an element in the solution set.
    list of lists L_rounds -- each element is a list containing the solution set L at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in L_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    '''
    check_inputs(objective, k)

    n = len(objective.groundset)
    queries_vec = []
    time_vec = []
    f_vec = []

    k_vals_vec_prog = []
    queries = 0
    comm.barrier()
    p_start = MPI.Wtime()

    L = []
    N = [ele for ele in objective.groundset]

    L_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]

    for i in range(k):

        # Compute the marginal addition for each elem in N, then add the best one to solution L; 
        # Then remove it from remaining elements N
        ele_vals_global_vals_max, ele_vals_global_max_ele = parallel_margvals_faster(objective, L, N, comm, rank, size)
        queries += len(N)
        L.append( ele_vals_global_max_ele )
        N.remove( ele_vals_global_max_ele )
        L_rounds.append([ele for ele in L])
        time_rounds.append(MPI.Wtime() -  p_start)
        query_rounds.append(queries)

        # Save data progressively
        if rank == p_root:
            if i%25==0:
                print( 'greedy_parallel round', i, 'of', k )
            if progsave_data and (i+1 in k_vals_vec):
                f_vec.append(objective.value(L))
                queries_vec.append(queries)
                time_vec.append(MPI.Wtime()-p_start)
                k_vals_vec_prog.append(i+1)
                np.savetxt(filepath_string + experiment_string +'_'+ algostring +".csv", \
                    np.vstack((f_vec, queries_vec, time_vec, k_vals_vec_prog)).T, delimiter=",")

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    #queries = np.float(k)*(n+n-(k-1))/2.0

    if rank == p_root:
        print ('greedy_parallel:', objective.value(L), queries, time)
    return objective.value(L), queries, time, L, L_rounds, time_rounds, query_rounds




# def greedy_parallel(objective, k, comm, rank, size, progsave_data=False, k_vals_vec=[], filepath_string='', \
#        experiment_string='', algostring='', p_root=0):

#     ''' 
#     (slightly slower implementation due to parallel architecture -- see alternative implementation above)
#     Accelerated lazy greedy algorithm: for k steps (Minoux 1978).
#     **NOTE** solution sets and values may be different than those found by our Greedy implementation, as the two implementations
#     break ties differently (i.e. when two elements have the same marginal value, the two implementations may not pick the same element to add)
#     
#     INPUTS:
#     class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
#     int k -- the cardinality constraint
#     comm -- the MPI4py Comm (MPI.COMM_WORLD)
#     int rank -- the processor's rank (comm.Get_rank())
#     int size -- the number of processors (comm.Get_size())

#     OPTIONAL INPUTS:
#     bool progsave_data -- if True, then the algorithm saves data after each round listed in the list k_vals_vec
#     list of ints k_vals_vec -- see above
#     string filepath_string -- used to make filenames if using progsave_data
#     string experiment string -- used to make filenames if using progsave_data
#     string algostring -- used to make filenames if using progsave_data

#     OUTPUTS:
#     float f(L) -- the value of the solution
#     int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
#     float time -- the processing time to optimize the function.
#     list L -- the solution, where each element in the list is an element in the solution set.
#     '''

#     n = len(objective.groundset)
#     f_vec = []
#     queries_vec = []
#     time_vec = []
#     k_vals_vec_prog = []
#     queries = 0
#     comm.barrier()
#     p_start = MPI.Wtime()

#     # All processors need to know current solution at each iter in order to evaluate marginal val of new elements
#     L = []
#     N = [ele for ele in objective.groundset]

#     #print len(N)   
#     for i in range(k):
#         # Compute the marginal addition for each elem in N, then add the best one to solution L; 
          # then remove it from remaining elements N
#         ele_vals_global_vals_max, ele_vals_global_max_ele = parallel_margvals(objective, L, N, comm, rank, size)
#         queries += len(N)        
#         L.append( ele_vals_global_max_ele )
#         N.remove( ele_vals_global_max_ele )

#         # Save data progressively
#         if rank == p_root:
#             if i%25==0:
#                 print( i, 'of', k)
#             if i+1 in k_vals_vec:
#                 f_vec.append(objective.value(L))

#                 queries_vec.append(queries)
#                 time_vec.append(MPI.Wtime()-p_start)
#                 k_vals_vec_prog.append(i+1)
#                 np.savetxt(filepath_string + experiment_string +'_'+ algostring +".csv", \
#                    np.vstack((f_vec, queries_vec, time_vec, k_vals_vec_prog)).T, delimiter=",")

#     comm.barrier()
#     p_stop = MPI.Wtime()
#     time = (p_stop - p_start)
#     #queries = np.float(k)*(n+n-k)/2.0
#     return objective.value(L), queries, time, L




def exhaustivemax(objective, k, eps, d, seed=42, m_reducedmean=False):
    '''
    Fahrbach et al. Algorithm 3 from
    "Submodular Maximization with Nearly Optimal Approximation, Adaptivity and Query Complexity"
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    float d -- the probability of failing to get a solution within the error tolerance

    OPTIONAL INPUTS:
    int seed -- random seed to use when drawing samples
    int or bool (False) m_reducedmean -- setting this to an int overrides the sample complexity m in reducedmean() subroutine

    OUTPUTS:
    float f(S) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list R -- the solution, where each element in the list is an element in the solution set.
    '''
    check_inputs(objective, k)


    time0 = datetime.now()
    queries = 0
    randstate = np.random.RandomState(seed)
    #queries_ideal = 0
    # Deltastar is the index of the best singleton element. 
    # CAREFUL ITS CAPITAL DELTA in writeup there is also lowercase that is different
    #Deltastar =  np.argmax([objective.value([ele]) for ele in objective.groundset])
    Deltastar =  np.max([objective.value([ele]) for ele in objective.groundset])

    queries += len(objective.groundset)
    #queries_ideal += len(objective.groundset)

    r = int( np.ceil( 2.0*np.log(k)/eps ) )

    #r = 4

    m = int( np.ceil( np.log(4.0)/eps ) )
    dhat = np.float(d) / ( np.float(r)*(np.float(m)+1) )
    R = []

    tau_vec = np.unique( [ (Deltastar/np.float(k)) * (1.0+eps)**i  for i in range(r) ] )

    for i, tau in enumerate(tau_vec):
        print ('\nexhaustivemax outer iter', i, 'of', len(tau_vec), ' == number of *unique* taus given r==', r)
        #tau = (Deltastar/np.float(k)) * (1.0+eps)**i # careful little tau different than T
        print('M REDUCEDMEAN', m_reducedmean)

        S = []

        for j in range(m):
            #print ('exhaustivemax inner iter', j, 'of', m, '==m')

            T, queries_threshsample = threshsampling(objective, (k-len(S)), tau*(1.0-eps)**j, eps, dhat, m_reducedmean, randstate)
            queries += queries_threshsample
            #queries_ideal += queries_threshsample_ideal

            S = list(set().union(S, T))
            if len(S) >= k:
                break

        f_S = objective.value(S)
        f_R = objective.value(R)
        queries += 1 # could technically do the above in one new call by keeping track of a prev_f_R variable
        #queries_ideal += 1
        if f_S > f_R:
            R = [ele for ele in S]
            print('new best value for R: ', f_S, 'len(R)=', len(R)) # note we want f_S here not f_R as R <- S!

    time = (datetime.now() - time0).total_seconds()
    return objective.value(R), queries, time, R




def threshsampling(objective, k, tau, eps, d, m_reducedmean, randstate):
    ''' Fahrbach et al. Algorithm 1 Threshold Sampling (subroutine in exhaustivemax())'''
    queries_threshsample = 0
    #queries_threshsample_ideal = 0
    eps_hat = eps/3.0
    r_logbase = 1.0 / (1.0-eps_hat)
    r_logargument = (2.0/d) * len(objective.groundset)
    r = int( np.ceil( (np.log(r_logargument) / np.log(r_logbase)) ) )

    m = int( np.ceil( (np.log(np.float(k)/eps_hat) ) ) )
    d_hat = d / ( 2.0*np.float(r)*( np.float(m)+1 ) )
    S = []
    A = [ele for ele in objective.groundset]

    for _ in range(r):
        A_filtered = [ele for ele in list(set(A)-set(S)) if objective.marginalval([ele], S) >= tau]
        A = A_filtered
        queries_threshsample += len(A)+1
        #queries_threshsample_ideal += len(A)+1
        print( 'threshsampling outer iter', _, 'of', r, '==r, len(A_filtered)=', len(A), 'high val elements')

        if not len(A):
            break

        t_option1_vec = np.unique([ int( np.floor( (1.0+eps_hat)**i ) ) for i in range(m) ])
        t_vec = np.minimum(t_option1_vec, len(A)*np.ones(len(t_option1_vec))).astype(np.int)
        for i, t in enumerate(t_vec):
            #print( 'threshsampling inner iter', i, 'of', m, '==m')
            # t_option1 = int( np.floor( (1.0+eps_hat)**i ) )
            # t = min(t_option1, len(A))
            #print( 't', t, 'is the size of the set we will try to add to S')
            reduced_mean_bool, queries_reduced_mean = \
                reducedmean(objective, S, A, t, eps_hat, d_hat, tau, m_reducedmean, randstate)

            queries_threshsample += queries_reduced_mean
            #queries_threshsample_ideal += queries_reduced_mean_ideal
            if reduced_mean_bool:
                print( 'we succeeded in reducedmean and can add a set of size t to S')
                break

        T = sampleX_wRepl(A, min(t, (k-len(S))))
        S = list(set().union(S, T))
        #print ('threshsampling now has len(S)=', len(S), 'f(S)=', objective.value(S))

        if len(S)>=k:
            print('exiting threshsampling due to having found a solution set of k elements')
            break
    return S, queries_threshsample#, queries_threshsample_ideal




def reducedmean(objective, S, A, t, eps, d, tau, m_reducedmean, randstate):
    '''
    Fahrbach et al. reduced mean subroutine from
    "Submodular Maximization with Nearly Optimal Approximation, Adaptivity and Query Complexity"
    '''
    queries_reduced_mean = 0 

    # If we did not manually set a smaller m_reducedmean to speed up computation (by sacrificing approx. guarantee), 
    # then set theoretical one
    if not m_reducedmean:
        m_reducedmean = int( 16.0 * np.ceil(   np.log(2.0/d)/eps**2   ) )

    # Each processor draws m/size samples, gets local avg surviving ele count, then reduce to global
    sample_sets = [sampleX_wRepl_randstate(A, t, randstate) for ss in range(int(np.ceil(m_reducedmean)))]
    sample_sets = [sset for sset in sample_sets if len(list(set(A) - set(sset)))] #ignore sets that dont add elements
    sample_elements = [randstate.choice(list(set(A) - set(sset))) for sset in sample_sets]

    if len(sample_elements):
        frac_surviving_ele = (1.0/len(sample_elements))*len( [ele for idx, ele in enumerate(sample_elements) if \
                                        objective.marginalval([ele], list(set().union(S, sample_sets[idx]))) > tau] )
    else:
        return True, queries_reduced_mean

    queries_reduced_mean +=  m_reducedmean

    if frac_surviving_ele <= (1.0-1.5*eps):
        return True, queries_reduced_mean
    else:
        return False, queries_reduced_mean



def reducedmean_parallel(objective, S, A, t, eps, d, tau, m_reducedmean, local_randstate, comm, rank, size):
    '''
    Fahrbach et al. reduced mean subroutine from
    "Submodular Maximization with Nearly Optimal Approximation, Adaptivity and Query Complexity"
    '''
    queries_reduced_mean = 0 

    # If we did not manually set a smaller m_reducedmean to speed up computation (by sacrificing approx. guarantee), 
    # then set theoretical one
    if not m_reducedmean:
        m_reducedmean = int( 16.0 * np.ceil(   np.log(2.0/d)/eps**2   ) )
    #print('samples:', int( 16.0 * np.ceil(   np.log(2.0/d)/eps**2   ) ))


    # # Root processor draws the samples and sends to all processors
    # sample_sets = None
    # if rank == p_root:
    #     sample_sets = [sampleX_wRepl_randstate(A, t, randstate) for ss in range(m)]
    # sample_sets = comm.bcast(sample_sets, p_root)

    # Each processor draws m/size samples, gets local avg surviving ele count, then reduce to global
    comm.barrier()
    sample_sets_local = [sampleX_wRepl_randstate(A, t, local_randstate) for ss in range(int(np.ceil(m_reducedmean/np.float(size))))]
    sample_sets_local = [sset for sset in sample_sets_local if len(list(set(A) - set(sset)))]
    sample_elements_local = [local_randstate.choice(list(set(A) - set(sset))) for sset in sample_sets_local]

    if len(sample_elements_local):
        frac_surviving_ele_local = (1.0/len(sample_elements_local))*len( [ele for idx, ele in enumerate(sample_elements_local) if \
                                        objective.marginalval([ele], list(set().union(S, sample_sets_local[idx]))) > tau] )
    else:
        frac_surviving_ele_local = []
    
    #assert(avg_len_surviving_ele_local0 == avg_len_surviving_ele_local)
    frac_surviving_ele_sum = comm.allreduce(frac_surviving_ele_local, op=MPI.SUM)
    #print('frac_surviving_ele_sum', frac_surviving_ele_sum, 'len(A)', len(A))

    # Make sure at least one sample returned (note that this fails when A contains a single element for example)
    if not isinstance(frac_surviving_ele_sum, (list,)):
       frac_surviving_ele = (1.0/size)*frac_surviving_ele_sum

    else:
        return True, queries_reduced_mean

    queries_reduced_mean +=  m_reducedmean

    if frac_surviving_ele <= (1.0-1.5*eps):
        return True, queries_reduced_mean
    else:
        return False, queries_reduced_mean



def exhaustivemax_parallel(objective, k, eps, d, comm, rank, size, p_root=0, randseed=42, m_reducedmean=False):
    '''
    Parallel implementation of Fahrbach et al. Algorithm 3 from
    "Submodular Maximization with Nearly Optimal Approximation, Adaptivity and Query Complexity"
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    float d -- the probaility that the algorithm will fail to find a solution within the error tolerance
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    
    OPTIONAL INPUTS:
    int p_root -- rank of master processor
    int randseed -- random seed to use when drawing samples
    int or bool (False) m_reducedmean -- setting this to an int overrides the sample complexity m in reducedmean() subroutine
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list R -- the solution, where each element in the list is an element in the solution set.
    '''
    check_inputs(objective, k)

    # Each processor seeds a UNIQUE random state (so each can draw a different set of samples!)
    local_randstate = np.random.RandomState(rank+randseed)
    global_randstate = np.random.RandomState(randseed)

    comm.barrier()
    p_start = MPI.Wtime()
    queries = 0

    Deltastar, _ = parallel_vals_faster(objective, [ele for ele in objective.groundset], comm, rank, size)
    queries += len(objective.groundset)

    r = int( np.ceil( 2.0*np.log(k)/eps ) )
    m = int( np.ceil( np.log(4.0)/eps ) )
    dhat = np.float(d) / ( np.float(r)*(np.float(m)+1) )
    R = []

    tau_vec = np.unique( [ (Deltastar/np.float(k)) * (1.0+eps)**i  for i in range(r) ] )

    for i, tau in enumerate(tau_vec):
        if rank == p_root:
            print ('\nexhaustivemax outer iter', i, 'of', len(tau_vec), ' == number of *unique* taus given r==', r)

        #tau = (Deltastar/np.float(k)) * (1.0+eps)**i # careful little tau different than T
        S = []

        for j in range(m):
            comm.barrier()
            # if rank == p_root:
            #     print ('exhaustivemax inner iter', j, 'of', m, '==m')

            T, queries_threshsample, queries_threshsample_ideal = threshsampling_parallel(objective, (k-len(S)), tau*(1.0-eps)**j, \
                                                                                            eps, dhat, m_reducedmean, \
                                                                                            local_randstate, \
                                                                                            global_randstate, \
                                                                                            comm=comm, \
                                                                                            rank=rank, \
                                                                                            size=size, \
                                                                                            p_root=p_root)
            queries += queries_threshsample

            S = list(set().union(S, T))
            if len(S) >= k:
                break

        f_S = objective.value(S)
        f_R = objective.value(R)
        queries += 1 # could technically do the above in one new call by keeping track of a prev_f_R variable
        #queries_ideal += 1
        if f_S > f_R:
            R = [ele for ele in S]
            if rank == p_root:
                print('new best value for R: ', f_S, 'len(R)=', len(R)) # note we want f_S here not f_R as R <- S!

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    return objective.value(R), queries, time, R




def binsearchmax_parallel(objective, k, eps, d, comm, rank, size, p_root=0, randseed=42, m_reducedmean=False):
    '''
    Parallel implementation of Fahrbach et al. Algorithm 4 from
    "Submodular Maximization with Nearly Optimal Approximation, Adaptivity and Query Complexity"
    (Modified for case when OPT is known so we don't need to iterate over multiple OPT guesses)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    float d -- the probaility that the algorithm will fail to find a solution within the error tolerance
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    
    OPTIONAL INPUTS:
    int p_root -- rank of master processor
    int randseed -- random seed to use when drawing samples
    int or bool (False) m_reducedmean -- setting this to an int overrides the sample complexity m in reducedmean() subroutine
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list R -- the solution, where each element in the list is an element in the solution set.
    '''

    check_inputs(objective, k)

    # Each processor seeds a UNIQUE random state (so each can draw a different set of samples!)
    local_randstate = np.random.RandomState(rank+randseed)
    global_randstate = np.random.RandomState(randseed)

    comm.barrier()
    p_start = MPI.Wtime()
    queries = 0

    Deltastar, _ = parallel_vals_faster(objective, [ele for ele in objective.groundset], comm, rank, size)
    queries += len(objective.groundset)

    L = np.float(Deltastar)
    U = np.float(k) * Deltastar

    if U/L >= 2.0*(10**6)/(d**2):
        print('\nPREPROCESSING NECESSARY (not implemented')
        raise Exception

    p = 1.0/np.log(np.float(k))
    m = np.floor (np.log(np.log(np.float(k)))/np.log(2.0))

    deltahatbinsearch = d/(m+1.0)

    for iii in range(int(m)):
        tau = (1.0/k) * np.sqrt(L * U/(2.0*p) )
        S, queries_threshsample, queries_threshsample_ideal = threshsampling_parallel(objective, k, tau, \
                                                                                            (1.0-p), deltahatbinsearch, m_reducedmean, \
                                                                                            local_randstate,\
                                                                                            global_randstate,\
                                                                                            comm=comm, \
                                                                                            rank=rank, \
                                                                                            size=size, \
                                                                                            p_root=p_root)

        if len(S) < k and objective.value(S) <= k*tau:
            U = 2.0*k*tau
        else:
            L = p*k*tau


    # Now run modified exhaustivemax
    r = int( np.ceil( 2.0*np.log(k)/eps ) )
    m = int( np.ceil( np.log(4.0)/eps ) )
    dhat = np.float(d) / ( np.float(r)*(np.float(m)+1) )
    R = []

    tau_vec_exmax_orig = np.unique( [ (Deltastar/np.float(k)) * (1.0+eps)**i  for i in range(r) ] )

    tau_vec = np.unique( [ (L/np.float(k)) * (1.0+eps)**i  for i in range(r) ] )
    tau_vec = tau_vec[tau_vec<= U/np.float(k)]

    # if rank == p_root:
    #     print ('len_tau_vec_exmax_orig', len(tau_vec_exmax_orig), 'tau_vec_exmax_orig', tau_vec_exmax_orig)
    #     print ('len_tau_vec', len(tau_vec), 'tau_vec', tau_vec)
    # print('Deltastar', Deltastar, 'r', r)
    # print('tau_vec', tau_vec)
    # comm.barrier()
    # ghghg

    for i, tau in enumerate(tau_vec):
        if rank == p_root:
            print ('\nbinsearchmax outer iter', i, 'of', len(tau_vec), ' == number of *unique* taus given r==', r)

        #tau = (Deltastar/np.float(k)) * (1.0+eps)**i # careful little tau different than T
        S = []

        for j in range(m):
            # if rank == p_root:
            #     print ('exhaustivemax inner iter', j, 'of', m, '==m')

            T, queries_threshsample, queries_threshsample_ideal = threshsampling_parallel(objective, (k-len(S)), tau*(1.0-eps)**j, \
                                                                                            eps, dhat, m_reducedmean, \
                                                                                            local_randstate, \
                                                                                            global_randstate, \
                                                                                            comm=comm, \
                                                                                            rank=rank, \
                                                                                            size=size, \
                                                                                            p_root=p_root)
            queries += queries_threshsample

            S = list(set().union(S, T))
            if len(S) >= k:
                break

        f_S = objective.value(S)
        f_R = objective.value(R)
        queries += 1 # could technically do the above in one new call by keeping track of a prev_f_R variable
        #queries_ideal += 1
        if f_S > f_R:
            R = [ele for ele in S]
            if rank == p_root:
                print('new best value for R: ', f_S, 'len(R)=', len(R)) # note we want f_S here not f_R as R <- S!

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    return objective.value(R), queries, time, R




def binsearchmax1OPT_parallel(objective, k, eps, d, comm, rank, size, OPT=None, p_root=0, randseed=42, m_reducedmean=False):
    '''
    MODIFIED VERSION OF:
    Parallel implementation of Fahrbach et al. Algorithm 4 from
    "Submodular Maximization with Nearly Optimal Approximation, Adaptivity and Query Complexity"
    (Modified for case when OPT is known so we don't need to iterate over multiple OPT guesses)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    float d -- the probaility that the algorithm will fail to find a solution within the error tolerance
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    
    OPTIONAL INPUTS:
    float OPT -- value of optimal solution. If left to none we will use sum of values of top k singletons.
    int p_root -- rank of master processor
    int randseed -- random seed to use when drawing samples
    int or bool (False) m_reducedmean -- setting this to an int overrides the sample complexity m in reducedmean() subroutine
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list R -- the solution, where each element in the list is an element in the solution set.
    list of lists R_rounds -- each element is a list containing the solution set R at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in R_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    '''
    check_inputs(objective, k)

    # Each processor seeds a UNIQUE random state (so each can draw a different set of samples!)
    local_randstate = np.random.RandomState(rank+randseed)
    global_randstate = np.random.RandomState(randseed)

    comm.barrier()
    p_start = MPI.Wtime()
    queries = 0

    # Here we modify algo to use a single better guess of OPT if we don't already have one
    if OPT is None:
        allvals = parallel_margvals_returnvals(objective, [], [ele for ele in objective.groundset], comm, rank, size)
        queries += len(objective.groundset)
        OPT = np.sum( np.sort(allvals)[-k:] )

    # Deltastar, _ = parallel_vals_faster_returnvals(objective, [ele for ele in objective.groundset], comm, rank, size)
    # queries += len(objective.groundset)

    # L = np.float(Deltastar)
    # U = np.float(k) * Deltastar

    # if U/L >= 2.0*(10**6)/(d**2):
    #     print('\nPREPROCESSING NECESSARY (not implemented')
    #     raise Exception

    # p = 1.0/np.log(np.float(k))
    # m = np.floor (np.log(np.log(np.float(k)))/np.log(2.0))

    # deltahatbinsearch = d/(m+1.0)

    # for iii in range(int(m)):
    #     tau = (1.0/k) * np.sqrt(L * U/(2.0*p) )
    #     S, queries_threshsample, queries_threshsample_ideal = threshsampling_parallel(objective, k, tau, \
    #                                                                                         (1.0-p), deltahatbinsearch, m_reducedmean, \
    #                                                                                         local_randstate, comm, rank, \
    #                                                                                         size, p_root)

    #     if len(S) < k and objective.value(S) <= k*tau:
    #         U = 2.0*k*tau
    #     else:
    #         L = p*k*tau




    # Now run modified exhaustivemax
    r = int( np.ceil( 2.0*np.log(k)/eps ) )
    m = int( np.ceil( np.log(4.0)/eps ) )
    dhat = np.float(d) / ( np.float(r)*(np.float(m)+1) )
    R = []
    R_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]

    #tau_vec_exmax_orig = np.unique( [ (Deltastar/np.float(k)) * (1.0+eps)**i  for i in range(r) ] )

    # tau_vec = np.unique( [ (L/np.float(k)) * (1.0+eps)**i  for i in range(r) ] )
    # tau_vec = tau_vec[tau_vec<= U/np.float(k)]

    # if rank == p_root:
    #     print ('len_tau_vec_exmax_orig', len(tau_vec_exmax_orig), 'tau_vec_exmax_orig', tau_vec_exmax_orig)
    #     print ('len_tau_vec', len(tau_vec), 'tau_vec', tau_vec)
    # print('Deltastar', Deltastar, 'r', r)
    # print('tau_vec', tau_vec)
    # comm.barrier()

    output_S_rounds_and_time_rounds=True # for threshsampling_parallel subroutine
    for i, tau in enumerate( [OPT/np.float(k)] ):
        if rank == p_root:
            print ('\nbinsearchmax outer iter', i, 'of', 1, ' == number of *unique* taus given r==', r)

        #tau = (Deltastar/np.float(k)) * (1.0+eps)**i # careful little tau different than T
        S = []

        for j in range(m):
            # if rank == p_root:
            #     print ('exhaustivemax inner iter', j, 'of', m, '==m')

            T, queries_threshsample, queries_threshsample_ideal, R_Rounds, time_rounds, query_rounds = \
                                                            threshsampling_parallel(objective, \
                                                            (k-len(S)), tau*(1.0-eps)**j, \
                                                            eps, dhat, m_reducedmean, \
                                                            local_randstate, \
                                                            global_randstate, \
                                                            comm=comm, \
                                                            rank=rank, \
                                                            size=size, \
                                                            p_root=p_root, \
                                                            S_rounds=R_rounds, \
                                                            time_rounds=time_rounds, \
                                                            query_rounds=query_rounds, \
                                                            p_start=p_start, \
                                                            output_S_rounds_and_time_rounds=output_S_rounds_and_time_rounds)
            queries += queries_threshsample

            S = list(set().union(S, T))

            if len(S) >= k:
                break

        f_S = objective.value(S)
        f_R = objective.value(R)
        queries += 1 # could technically do the above in one new call by keeping track of a prev_f_R variable
        R_rounds.append( S )
        time_rounds.append( MPI.Wtime() - p_start )
        query_rounds.append( queries )

        if f_S > f_R:
            R = [ele for ele in S]
            if rank == p_root:
                print('new best value for R: ', f_S, 'len(R)=', len(R)) # note we want f_S here not f_R as R <- S!

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    return objective.value(R), queries, time, R, R_Rounds, time_rounds, query_rounds




def threshsampling_parallel(objective, k, tau, eps, d, m_reducedmean, local_randstate, global_randstate, comm, rank, size, p_root, \
                            S_rounds=[[]], time_rounds=[0], query_rounds=[0], p_start=0, output_S_rounds_and_time_rounds=False):
    '''
    Fahrbach et al. threshold sampling subroutine from
    "Submodular Maximization with Nearly Optimal Approximation, Adaptivity and Query Complexity"
    '''
    prev_queries = query_rounds[-1]
    prev_S = S_rounds[-1]
    queries_threshsample = 0
    queries_threshsample_ideal = 0
    eps_hat = eps/3.0
    r_logbase = 1.0 / (1.0-eps_hat)
    r_logargument = (2.0/d) * len(objective.groundset)
    r = int( np.ceil( (np.log(r_logargument) / np.log(r_logbase)) ) )
    #r = 20

    m = int( np.ceil( (np.log(np.float(k)/eps_hat) ) ) )
    d_hat = d / ( 2.0*np.float(r)*( np.float(m)+1 ) )
    S = []
    A = [ele for ele in objective.groundset]

    for _ in range(r):
        A_filtered = parallel_margvals_returneles_ifthresh(objective, S, list(set(A)-set(S)), tau, comm, rank, size)
        A = A_filtered
        queries_threshsample += len(A)+1
        queries_threshsample_ideal += len(A)+1
        # if rank == p_root:
        #     print( 'threshsampling main iter', _, 'of', r, '==r, len(A_filtered)=', len(A), 'high val elements', 'S', S, 'lenA', len(A))

        if not len(A):
            break

        t_option1_vec = np.unique([ int( np.floor( (1.0+eps_hat)**i ) ) for i in range(m) ])
        t_vec = np.minimum(t_option1_vec, len(A)*np.ones(len(t_option1_vec))).astype(np.int)
        for i, t in enumerate(t_vec):

            # t_option1 = int( np.floor( (1.0+eps_hat)**i ) )
            # t = min(t_option1, len(A))

            # if rank == p_root:
            #     print( 'idx', i, 't', t, 'is the size of the set we will try to add to S')
            reduced_mean_bool, queries_reduced_mean = reducedmean_parallel(objective, S, A, t, eps_hat, d_hat, tau, \
                                                            m_reducedmean, local_randstate, comm, rank, size)
            # print(i, 'THRESHSAMPLING PARALLEL POST REDMEAN RANK', rank, 'reduced_mean_bool', reduced_mean_bool)
            queries_threshsample += queries_reduced_mean
            if reduced_mean_bool:
                # if rank == p_root:
                #     print( 'we succeeded in reducedmean and can add a set of size t to S')
                break

        T = sampleX_wRepl_randstate(A, min(t, (k-len(S))), global_randstate)

        S = list(set().union(S, T))

        S_rounds.append([ele for ele in set().union(prev_S, S)])
        time_rounds.append( MPI.Wtime() - p_start )
        # print(query_rounds, queries_threshsample)
        query_rounds.append(prev_queries+queries_threshsample)

        # if rank == p_root:
        #     print ('threshsampling now has len(S)=', len(S), 'f(S)=', objective.value(S))

        if len(S)>=k:
            break

    if output_S_rounds_and_time_rounds:
        return S, queries_threshsample, queries_threshsample_ideal, S_rounds, time_rounds, query_rounds

    return S, queries_threshsample, queries_threshsample_ideal




def reducedmean_parallel(objective, S, A, t, eps, d, tau, m_reducedmean, local_randstate, comm, rank, size):
    '''
    Fahrbach et al. reduced mean subroutine from
    "Submodular Maximization with Nearly Optimal Approximation, Adaptivity and Query Complexity"
    '''
    comm.barrier()
    queries_reduced_mean = 0 

    # If we did not manually set a smaller m_reducedmean to speed up computation (by sacrificing approx. guarantee), 
    # then set theoretical one
    if not m_reducedmean:
        m_reducedmean = int( 16.0 * np.ceil(   np.log(2.0/d)/eps**2   ) )
    #print('samples:', int( 16.0 * np.ceil(   np.log(2.0/d)/eps**2   ) ))


    # # Root processor draws the samples and sends to all processors
    # sample_sets = None
    # if rank == p_root:
    #     sample_sets = [sampleX_wRepl_randstate(A, t, randstate) for ss in range(m)]
    # sample_sets = comm.bcast(sample_sets, p_root)

    # Each processor draws m/size samples, gets local avg surviving ele count, then reduce to global
    comm.barrier()
    sample_sets_local = [sampleX_wRepl_randstate(A, t, local_randstate) for ss in range(int(np.ceil(m_reducedmean/np.float(size))))]
    sample_sets_local = [sset for sset in sample_sets_local if len(list(set(A) - set(sset)))]
    sample_elements_local = [local_randstate.choice(list(set(A) - set(sset))) for sset in sample_sets_local]

    if len(sample_elements_local):
        frac_surviving_ele_local = (1.0/len(sample_elements_local))*len( [ele for idx, ele in enumerate(sample_elements_local) if \
                                        objective.marginalval([ele], list(set().union(S, sample_sets_local[idx]))) > tau] )
    else:
        frac_surviving_ele_local = []
    
    #assert(avg_len_surviving_ele_local0 == avg_len_surviving_ele_local)
    frac_surviving_ele_sum = comm.allreduce(frac_surviving_ele_local, op=MPI.SUM)
    #print('frac_surviving_ele_sum', frac_surviving_ele_sum, 'len(A)', len(A))

    # Make sure at least one sample returned (note that this fails when A contains a single element for example)
    if not isinstance(frac_surviving_ele_sum, (list,)):
       frac_surviving_ele = (1.0/size)*frac_surviving_ele_sum

    else:
        return True, queries_reduced_mean

    queries_reduced_mean +=  m_reducedmean

    if frac_surviving_ele <= (1.0-1.5*eps):
        return True, queries_reduced_mean
    else:
        return False, queries_reduced_mean




def FAST_knowopt(objective, k, eps, OPT, preprocess_add=False, \
                            lazy_binsearch=False, lazyouterX=False, debug_asserts=False, \
                            weight_sampling_eps=1.0, sample_threshold=False, lazyskipbinsearch=False, \
                            allvals=None, OPT_guess_count=1, verbose=True, seed=42):

    '''
    FAST Algorithm for SUbmodular Mazimization, VERSION FOR CASE WHERE OPT IS KNOWN (Breuer, Balkanski, Singer 2019)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    float OPT -- value of the optimal solution

    OPTIONAL INPUTS:
    bool preprocess_add: turns on/off the 'Preprocessing  the  sequence' speedup
    bool lazy_binsearch: turns on/off lazy updates within the binary search for i^*
    bool lazyouterX: turns on/off lazy updates in outer while loop for X
    bool debug_asserts: turns on/off many debugging assert statements to debug parallel implementations of various functions
    float weight_sampling_eps: change from 1.0 to reweight different epsilons in algo separately. Increasing accelerates binseach for i^*
    bool sample_threshold: turns on/off sampling elements within the binary search for i^* (vs. trying them all)
    bool lazyskipbinsearch: turns on/off the speedup where if preprocessing is successful we can skip binary search for i^*
    list of floats allvals: if you provide a list of the value f(e) for e in the groundset (in order of groundset), this accelerates the algo
    OPT_guess_count: index of current guess for OPT. If we know OPT or succeed at the 1st guess many fewer samples needed in R when searching i*.
    bool verbose: turns on/off many print statements
    int seed -- random seed to use when drawing samples

    OUTPUTS:
    float f(S) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list S -- the solution, where each element in the list is an element in the solution set.
    list of lists S_rounds -- each element is a list containing the solution set S at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in S_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    '''

    check_inputs(objective, k)

    n = len(objective.groundset)

    randstate = np.random.RandomState(seed)
    queries = 0
    time0 = datetime.now()

    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    iters = 0

    # Initialize lazyouterX
    if allvals is None:
        eles_latest_marg_Xdiscards = np.inf*np.ones(n) # Set to inf so we will check all elements a-priori
    else:
        eles_latest_marg_Xdiscards = np.copy(allvals)


    while len(S) < k and iters < np.ceil(1.0/eps):

        if OPT_guess_count == 1:
            sample_complexity = compute_sample_complexity_threshold(eps, k, weight_sampling_eps)
        else:
            sample_complexity = compute_sample_complexity_threshold_full(eps, k, weight_sampling_eps, n)

        if sample_threshold and verbose:
            print('\n')
            print('SAMPLE COMPLEXITY:', sample_complexity)
            print('\n')

        iters += 1

        t = (1.0-eps)*(OPT - objective.value(S))/np.float(k)
        X_lowval_to_ignore = np.where(eles_latest_marg_Xdiscards < t)[0]
        X = list(np.sort(list(set(objective.groundset) - set(S)))) 
        
        if lazyouterX:
            X = list(set(X) - set(X_lowval_to_ignore))
            if verbose:
                print('\n')
                print('LAZYOUTERX ignored ', len(X_lowval_to_ignore), 'Groundset Elements!', 't=', t)
            # print([objective.marginalval([ele], S) for ele in X_lowval_to_ignore])
            # print('eles are:', [ele for ele in X_lowval_to_ignore])


        while len(X) and len(S)<k:              

            prev_len_X = len(X)
            if preprocess_add:

                max_x_seq_size = int( np.min(( (k-len(S)), len(X)) ) )             
                x_seq = sampleX_randstate(X, max_x_seq_size, randstate)  
                S_len = len(S)
                [ S.append(ele) for ii, ele in enumerate(x_seq) if (len(S)<k) and \
                            (objective.marginalval( [ele],  list(np.sort(list(set().union(S, x_seq[:(ii)])))) ) >= t)]
                S_rounds.append([ele for ele in S])
                time_rounds.append( (datetime.now() - time0).total_seconds() ) 
                queries += len(x_seq)
                query_rounds.append(queries)
                print('ADDED', len(S)-S_len, 'NEW ELEMENTS IN PESSIMISTIC STEP')


            # Discard low value elements from X again (speeds up the runtime)
            new_X_setmin_S = list(np.sort(list(set(X)-set(S))))


            X_margvals =  [ objective.marginalval([ele], S) for ele in new_X_setmin_S ] 
            queries += len(new_X_setmin_S) + 1

            if len(new_X_setmin_S):
                #eles_latest_marg_Xdiscards[[np.array(new_X_setmin_S)]] = X_margvals
                eles_latest_marg_Xdiscards[np.array(new_X_setmin_S)] = X_margvals

            X = [ ele for idx, ele in enumerate(new_X_setmin_S) if X_margvals[idx] >= t]
            #X_margvals_filtered = [ margval for margval in X_margvals if margval >= t]
            #print('len(X_margvals)', len(X_margvals), 'len(X)', len(X), 'len(X_margvals_filtered)', len(X_margvals_filtered))
            queries += len(list(set(X)-set(S)))+1


            # If we added/discarded enough elements with the pessimistic add step, then 
            # skip the i* sequence search and move to the next round.
            #if lazyskipbinsearch and len(eles_to_add_to_S) >= eps*prev_len_X:
            if (not len(X)) or (lazyskipbinsearch and (len(X) <= (1.0-weight_sampling_eps*eps)*prev_len_X)):
                print('Lazy skipping binary search')
                continue

            else:
                max_a_seq_size = min((k-len(S)), len(X))                
                a_seq = sampleX_randstate(X, max_a_seq_size, randstate)  

                I2 = make_I2_0idx(eps, max_a_seq_size)
                I2_copy = np.copy(I2)

                i_star_found = False

                # Initialize lazy binary search
                eles_highest_i_passthresh = -np.inf*np.ones(n)
                eles_lowest_i_failthresh = np.inf*np.ones(n)

                # Initialize samples to determine whether we pass a particular threshold to check each i:
                num_samples = int(np.min((sample_complexity, len(X))))
                R = sampleX_randstate(X, num_samples, randstate) 

                while not i_star_found:
                    # Update binary search
                    midI2 =  len(I2) // 2
                    i = I2[midI2]
                    i_star_found = len(I2) == 1 # We do this in the while loop to make binary search check final element in I2
                

                    # Prepare to skip redundant computations of marginal values
                    S_u_aseq_i = np.sort(list(set().union(S, a_seq[:i])))
                    X_setmin_aseq_i = np.sort(list(set(X)-set(S_u_aseq_i)))


                    # Prepare just a subset of X that intersects with samples R, which we use to estimate |X_i| via sampling.
                    if sample_threshold:
                        X_setmin_aseq_i = list(np.sort( list(set(R).intersection(X_setmin_aseq_i)) ))
                    if verbose:
                        print('len(X_setmin_aseq_i)', len(X_setmin_aseq_i), 'sample_threshold=', sample_threshold)


                    # In binary search, we already know some elements exceed the threshold 
                    # or cannot exceed the threshold due to submodularity
                    X_setmin_aseq_i_knowntrue  = [ele for ele in X_setmin_aseq_i if eles_highest_i_passthresh[ele] > i]
                    X_setmin_aseq_i_knownfalse = [ele for ele in X_setmin_aseq_i if eles_lowest_i_failthresh[ele] < i]

                    X_setmin_aseq_i_totry = list(np.copy(X_setmin_aseq_i))
                    if lazy_binsearch:
                        X_setmin_aseq_i_totry = list(np.sort(list(set(X_setmin_aseq_i)\
                                                -set(X_setmin_aseq_i_knowntrue)\
                                                -set(X_setmin_aseq_i_knownfalse))))


                    # Compute X_i, the set of elements with high marginal contribution to the current sequence of length i-1
                    X_i_totry_margvals = [ objective.marginalval([ele], S_u_aseq_i) for ele in X_setmin_aseq_i_totry ]
                    X_i = [ele for idx, ele in enumerate(X_setmin_aseq_i_totry) if t <= X_i_totry_margvals[idx] ]
                    len_X_i_trues = np.float(len(X_i))
                    queries += len(X_setmin_aseq_i_totry)+1

                    # If computing |X_i| lazily, add the count of known high value elements
                    if lazy_binsearch: 
                        len_X_i_trues = len_X_i_trues + len(X_setmin_aseq_i_knowntrue)

                    if verbose:
                        print('len(X_i)', len(X_i), 'len(X)', len(X), '#knowntrue =',  len(X_setmin_aseq_i_knowntrue), \
                            '#knownfalse=', len(X_setmin_aseq_i_knownfalse), '#queried=', len(X_i_totry_margvals))

                    #print('len(I2) is', len(I2), 'len(X_i) is', len(X_i), 'len(X) is', len(X), 'len a_seq is', len(a_seq), \
                    #'JUST FINISHED checking for i in range',  I2[0], I2[-1] )
                    if verbose:
                        print('i', i, 'len_X_i_trues', len_X_i_trues, 'len(X_setmin_aseq_i)',\
                            len(X_setmin_aseq_i), '(1.0-eps)*len(X_setmin_aseq_i)', (1.0-eps)*len(X_setmin_aseq_i))

                    # X_i too small, so need to check the left half
                    #' THIS IS ME HERE SAYING WE NEED TO (DISCARD + ADD) A CONSTANT FRACTION EACH ITERATION'
                    #if len(X_i) < (1.0-eps)*len(X):
                    if (not i_star_found) and (len_X_i_trues < (1.0-2.0*eps)*len(X_setmin_aseq_i)):
                    #if len(X_i) < (1.0-eps)*len(X_setmin_aseq_i):
                        I2 = I2[:midI2]
                        # update lazily
                        if len(X_i):
                            #eles_highest_i_passthresh[[np.array(X_i)]] = np.maximum( i*np.ones(len(X_i)), \
                            #eles_highest_i_passthresh[[np.array(X_i)]] )
                            eles_highest_i_passthresh[np.array(X_i)] = \
                                np.maximum( i*np.ones(len(X_i)), eles_highest_i_passthresh[np.array(X_i)] )
                        if verbose:
                            print('len(I2) is', len(I2), 'len(X_i) is', len(X_i), 'len(X) is', len(X), ' looking left', \
                                'len a_seq is', len(a_seq), 'checking for i in range',  I2[0], I2[-1] )

                    # X_i big enough, so need to check the right half
                    elif (not i_star_found):
                        I2 = I2[midI2:]
                        # update lazily
                        if verbose:
                            print('X_setmin_aseq_i_knownfalse', X_setmin_aseq_i_knownfalse)
                        fail_eles = list(set(X_setmin_aseq_i_totry) - set(X_i))
                        if len(fail_eles):
                            #eles_lowest_i_failthresh[[(fail_eles)]] = np.minimum( i*np.ones(len(fail_eles)), \
                            #eles_lowest_i_failthresh[[np.array(fail_eles)]] ) 
                            eles_lowest_i_failthresh[(fail_eles)] = \
                                np.minimum( i*np.ones(len(fail_eles)), eles_lowest_i_failthresh[np.array(fail_eles)] ) 

                        if verbose:
                            print('should have updated a maximum of', len(X_setmin_aseq_i_totry) - len(X_i), \
                                'i values in eles_lowest_i_failthresh:', \
                                'np.sum(eles_lowest_i_failthresh==i)', np.sum(eles_lowest_i_failthresh==i))
                            print('len(I2) is', len(I2), 'len(X_i) is', len(X_i), 'len(X) is', len(X), ' looking right', \
                                'len a_seq is', len(a_seq), 'checking for i in range',  I2[0], I2[-1] )

                    elif (len_X_i_trues < (1.0-2.0*eps)*len(X_setmin_aseq_i)):
                        # I_star found.
                        # Final element Did not pass threshold, so final i_star is one left in I2 (or -1 if we are currently at i=0)
                        i_notpass_idx = np.where(I2[0] == I2_copy)[0][0]
                        if i_notpass_idx >0:
                            i_star = I2_copy[i_notpass_idx-1]
                        else:
                            i_star = -1

                        if verbose:
                            print('I2[0] did not pass', 'I2[0]=', I2[0], 'I2_copy[np.where(I2[0] == I2_copy)[0][0]]=',\
                                I2_copy[np.where(I2[0] == I2_copy)[0][0]], 'np.where(I2[0] == I2_copy)[0][0]=', \
                                np.where(I2[0] == I2_copy)[0][0], 'i_star=', i_star)
                    else:
                        # I_star found.
                        # Final element PASSED threshold, so final i_star is remaining element in I2
                        i_star = I2[0]

                    S_rounds.append([ele for ele in S])
                    time_rounds.append( (datetime.now() - time0).total_seconds() )  
                    query_rounds.append(queries)

                # Add the new sequence a_i_star to S
                a_seq_to_add = a_seq[:(i_star+1)]
                S = list(np.sort(list(set().union(S, a_seq_to_add))))
                S_rounds[-1] = [ele for ele in S]
                time_rounds[-1] = (datetime.now() - time0).total_seconds()
                query_rounds[-1] = queries

                if verbose:
                    print('i_star is', i_star, 'len a_seq is', len(a_seq), 'len(S)=', len(S))
                if len(S) == k:
                    break

                X = list(np.sort(list(set(X)-set(S))))

    #return S_star, I1_found
    time = (datetime.now() - time0).total_seconds()

    print ('Fast (NONparallel version):', objective.value(S), queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
    print('preprocess_add=', preprocess_add, 'lazy_binsearch=', lazy_binsearch, 'lazyouterX=', lazyouterX, \
        'debug_asserts=', debug_asserts, 'weight_sampling_eps=', weight_sampling_eps, 'sample_threshold=', sample_threshold)
    return objective.value(S), queries, time, S, S_rounds, time_rounds, query_rounds#, len_X_i_vec, a_seq_list#, iter_outer_loop, iter_while2, iter_while3, iter_while4



def FAST_knowopt_parallel(objective, k, eps, OPT, comm, rank, size, preprocess_add=False, \
                            lazy_binsearch=False, lazyouterX=False, debug_asserts=False, \
                            weight_sampling_eps=1.0, sample_threshold=False, lazyskipbinsearch=False, \
                            allvals=None, OPT_guess_count=1, verbose=True, p_root=0, seed=42):

    '''
    FAST Algorithm for Submodular Mazimization, VERSION FOR CASE WHERE OPT IS KNOWN (Breuer, Balkanski, Singer 2019)
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    float OPT -- value of the optimal solution
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    bool preprocess_add: turns on/off the 'Preprocessing  the  sequence' speedup
    bool lazy_binsearch: turns on/off lazy updates within the binary search for i^*
    bool lazyouterX: turns on/off lazy updates in outer while loop for X
    bool debug_asserts: turns on/off many debugging assert statements to debug parallel implementations of various functions
    float weight_sampling_eps: change from 1.0 to reweight different epsilons in algo separately. Increasing accelerates binseach for i^*
    bool sample_threshold: turns on/off sampling elements within the binary search for i^* (vs. trying them all)
    bool lazyskipbinsearch: turns on/off the speedup where if preprocessing is successful we can skip binary search for i^*
    list of floats allvals: if you provide a list of the value f(e) for e in the groundset (in order of groundset), this accelerates the algo
    OPT_guess_count: index of current guess for OPT. If we know OPT or succeed at the 1st guess many fewer samples needed in R when searching i*.
    bool verbose: turns on/off many print statements
    int seed -- random seed to use when drawing samples

    OUTPUTS:
    float f(S) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list S -- the solution, where each element in the list is an element in the solution set.
    list of lists S_rounds -- each element is a list containing the solution set S at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in S_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    '''

    check_inputs(objective, k)
    comm.barrier()
    p_start = MPI.Wtime()
    n = len(objective.groundset)

    randstate = np.random.RandomState(seed)
    queries = 0
    time0 = datetime.now()

    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    iters = 0

    # Initialize lazyouterX
    if allvals is None:
        eles_latest_marg_Xdiscards = np.inf*np.ones(n) # Set to inf so we will check all elements a-priori
    else:
        eles_latest_marg_Xdiscards = np.copy(allvals)


    while len(S) < k and iters < np.ceil(1.0/eps):

        if OPT_guess_count == 1:
            sample_complexity = compute_sample_complexity_threshold(eps, k, weight_sampling_eps)
        else:
            sample_complexity = compute_sample_complexity_threshold_full(eps, k, weight_sampling_eps, n)

        if sample_threshold and rank==p_root and verbose:
            print('\n')
            print('SAMPLE COMPLEXITY:', sample_complexity)
            print('\n')

        iters += 1

        t = (1.0-eps)*(OPT - objective.value(S))/np.float(k)
        X_lowval_to_ignore = np.where(eles_latest_marg_Xdiscards < t)[0]
        X = list(np.sort(list(set(objective.groundset) - set(S)))) 
        
        if lazyouterX:
            X = list(set(X) - set(X_lowval_to_ignore))
            if rank == p_root and verbose:
                print('\n')
                print('LAZYOUTERX ignored ', len(X_lowval_to_ignore), 'Groundset Elements!', 't=', t)
            # print([objective.marginalval([ele], S) for ele in X_lowval_to_ignore])
            # print('eles are:', [ele for ele in X_lowval_to_ignore])


        while len(X) and len(S)<k:              

            prev_len_X = len(X)
            if preprocess_add:

                max_x_seq_size = int( np.min(( (k-len(S)), len(X)) ) )             
                x_seq = sampleX_randstate(X, max_x_seq_size, randstate)  
                S_len = len(S)
                eles_to_add_to_S = parallel_pessimistically_add_x_seq(objective, S, x_seq, t, comm, rank, size)
                queries += len(x_seq)
                
                if debug_asserts:
                    eles_to_add_to_S0 = [ ele for ii, ele in enumerate(x_seq) \
                        if (len(S)<k) and (objective.marginalval( [ele],  list(np.sort(list(set().union(S, x_seq[:(ii)])))) ) >= t) ]
                    assert(np.array_equal(eles_to_add_to_S0, eles_to_add_to_S))

                [ S.append(ele) for ele in eles_to_add_to_S ]
                S_rounds.append([ele for ele in S])
                time_rounds.append( MPI.Wtime() - p_start ) 
                query_rounds.append(queries)

                if rank == p_root:
                    print('ADDED', len(S)-S_len, 'NEW ELEMENTS IN PESSIMISTIC STEP')


            # Discard low value elements from X again (speeds up the runtime)
            new_X_setmin_S = list(np.sort(list(set(X)-set(S))))


            X_margvals = parallel_margvals_returnvals(objective, S, new_X_setmin_S, comm, rank, size)
            queries += len(new_X_setmin_S) + 1

            if debug_asserts:
                X_margvals0 = [ objective.marginalval([ele], S) for ele in new_X_setmin_S ] 
                assert(np.array_equal(X_margvals0, X_margvals))

            if len(new_X_setmin_S):
                #eles_latest_marg_Xdiscards[[np.array(new_X_setmin_S)]] = X_margvals
                eles_latest_marg_Xdiscards[np.array(new_X_setmin_S)] = X_margvals

            X = [ ele for idx, ele in enumerate(new_X_setmin_S) if X_margvals[idx] >= t]
            #X_margvals_filtered = [ margval for margval in X_margvals if margval >= t]
            #print('len(X_margvals)', len(X_margvals), 'len(X)', len(X), 'len(X_margvals_filtered)', len(X_margvals_filtered))
            queries += len(list(set(X)-set(S)))+1


            # If we added/discarded enough elements with the pessimistic add step, then 
            # skip the i* sequence search and move to the next round.
            #if lazyskipbinsearch and len(eles_to_add_to_S) >= eps*prev_len_X:
            if (not len(X)) or (lazyskipbinsearch and (len(X) <= (1.0-weight_sampling_eps*eps)*prev_len_X)):
                if rank == p_root:
                    print('Lazy skipping binary search')
                continue

            else:
                max_a_seq_size = min((k-len(S)), len(X))                
                a_seq = sampleX_randstate(X, max_a_seq_size, randstate)  

                I2 = make_I2_0idx(eps, max_a_seq_size)
                I2_copy = np.copy(I2)

                i_star_found = False

                # Initialize lazy binary search
                eles_highest_i_passthresh = -np.inf*np.ones(n)
                eles_lowest_i_failthresh = np.inf*np.ones(n)

                # Initialize samples to determine whether we pass a particular threshold to check each i:
                num_samples = int(np.min((sample_complexity, len(X))))
                R = sampleX_randstate(X, num_samples, randstate) 

                while not i_star_found:
                    # Update binary search
                    midI2 =  len(I2) // 2
                    i = I2[midI2]
                    i_star_found = len(I2) == 1 # We do this in the while loop to make binary search check final element in I2
                

                    # Prepare to skip redundant computations of marginal values
                    S_u_aseq_i = np.sort(list(set().union(S, a_seq[:i])))
                    X_setmin_aseq_i = np.sort(list(set(X)-set(S_u_aseq_i)))


                    # Prepare just a subset of X that intersects with samples R, which we use to estimate |X_i| via sampling.
                    if sample_threshold:
                        X_setmin_aseq_i = list(np.sort( list(set(R).intersection(X_setmin_aseq_i)) ))
                    if rank == p_root and verbose:
                        print('len(X_setmin_aseq_i)', len(X_setmin_aseq_i), 'sample_threshold=', sample_threshold)


                    # In binary search, we already know some elements exceed the threshold 
                    # or cannot exceed the threshold due to submodularity
                    X_setmin_aseq_i_knowntrue  = [ele for ele in X_setmin_aseq_i if eles_highest_i_passthresh[ele] > i]
                    X_setmin_aseq_i_knownfalse = [ele for ele in X_setmin_aseq_i if eles_lowest_i_failthresh[ele] < i]

                    X_setmin_aseq_i_totry = list(np.copy(X_setmin_aseq_i))
                    if lazy_binsearch:
                        X_setmin_aseq_i_totry = list(np.sort(list(set(X_setmin_aseq_i)\
                                                -set(X_setmin_aseq_i_knowntrue)\
                                                -set(X_setmin_aseq_i_knownfalse))))


                    # Compute X_i, the set of elements with high marginal contribution to the current sequence of length i-1
                    X_i_totry_margvals = parallel_margvals_returnvals(objective, S_u_aseq_i, X_setmin_aseq_i_totry, comm, rank, size)
                    X_i = [ele for idx, ele in enumerate(X_setmin_aseq_i_totry) if t <= X_i_totry_margvals[idx] ]
                    len_X_i_trues = np.float(len(X_i))
                    queries += len(X_setmin_aseq_i_totry)+1

                    #print('rank', rank, 'len(X_setmin_aseq_i_totry)', len(X_setmin_aseq_i_totry), \
                    #'len(X_i_totry_margvals)', len(X_i_totry_margvals), 'len_X_i_trues0', len_X_i_trues0)
                    # len_X_i_trues = 0
                    # if len(X_setmin_aseq_i_totry):
                    #     len_X_i_trues = parallel_X_i_sampleestimate_sum(objective, S_u_aseq_i, X_setmin_aseq_i_totry, \
                    #t, sample_complexity, randstate, comm, rank, size)


                    if debug_asserts:
                        X_i_totry_margvals = [ objective.marginalval([ele], S_u_aseq_i) for ele in X_setmin_aseq_i_totry ]
                        X_i0 = [ele for idx, ele in enumerate(X_setmin_aseq_i_totry) if t <= X_i_totry_margvals[idx] ]
                        len_X_i_trues0 = np.float(len(X_i0))
                        assert(len_X_i_trues==len_X_i_trues0)
                        assert(np.array_equal(X_i0, X_i))


                    # If computing |X_i| lazily, add the count of known high value elements
                    if lazy_binsearch: 
                        len_X_i_trues = len_X_i_trues + len(X_setmin_aseq_i_knowntrue)


                    # Checks that lazy binary search matches non-lazy binary search exactly
                    if debug_asserts and not sample_threshold:
                        X_i_testing = [True for ele in X_setmin_aseq_i if t <= objective.marginalval([ele], S_u_aseq_i) ]
                        X_i_testing_eles = [ele for ele in X_setmin_aseq_i if t <= objective.marginalval([ele], S_u_aseq_i) ]
                        assert(len_X_i_trues==np.sum(X_i_testing))
                        assert( set(X_i_testing_eles) == set().union(X_i, X_setmin_aseq_i_knowntrue) )


                    if rank == p_root and verbose:
                        print('len(X_i)', len(X_i), 'len(X)', len(X), '#knowntrue =',  len(X_setmin_aseq_i_knowntrue), \
                            '#knownfalse=', len(X_setmin_aseq_i_knownfalse), '#queried=', len(X_i_totry_margvals))

                    #print('len(I2) is', len(I2), 'len(X_i) is', len(X_i), 'len(X) is', len(X), 'len a_seq is', len(a_seq), \
                    #'JUST FINISHED checking for i in range',  I2[0], I2[-1] )
                    if rank == p_root and verbose:
                        print('i', i, 'len_X_i_trues', len_X_i_trues, 'len(X_setmin_aseq_i)',\
                            len(X_setmin_aseq_i), '(1.0-eps)*len(X_setmin_aseq_i)', (1.0-eps)*len(X_setmin_aseq_i))

                    # X_i too small, so need to check the left half
                    #' WE NEED TO (DISCARD + ADD) A CONSTANT FRACTION EACH ITERATION'
                    #if len(X_i) < (1.0-eps)*len(X):
                    if (not i_star_found) and (len_X_i_trues < (1.0-2.0*eps)*len(X_setmin_aseq_i)):
                    #if len(X_i) < (1.0-eps)*len(X_setmin_aseq_i):
                        I2 = I2[:midI2]
                        # update lazily
                        if len(X_i):
                            #eles_highest_i_passthresh[[np.array(X_i)]] = np.maximum( i*np.ones(len(X_i)), \
                            #eles_highest_i_passthresh[[np.array(X_i)]] )
                            eles_highest_i_passthresh[np.array(X_i)] = \
                                np.maximum( i*np.ones(len(X_i)), eles_highest_i_passthresh[np.array(X_i)] )
                        if rank == p_root and verbose:
                            print('len(I2) is', len(I2), 'len(X_i) is', len(X_i), 'len(X) is', len(X), ' looking left', \
                                'len a_seq is', len(a_seq), 'checking for i in range',  I2[0], I2[-1] )

                    # X_i big enough, so need to check the right half
                    elif (not i_star_found):
                        I2 = I2[midI2:]
                        # update lazily
                        if rank == p_root and verbose:
                            print('X_setmin_aseq_i_knownfalse', X_setmin_aseq_i_knownfalse)
                        fail_eles = list(set(X_setmin_aseq_i_totry) - set(X_i))
                        if len(fail_eles):
                            #eles_lowest_i_failthresh[[(fail_eles)]] = np.minimum( i*np.ones(len(fail_eles)), \
                            #eles_lowest_i_failthresh[[np.array(fail_eles)]] ) 
                            eles_lowest_i_failthresh[(fail_eles)] = \
                                np.minimum( i*np.ones(len(fail_eles)), eles_lowest_i_failthresh[np.array(fail_eles)] ) 

                        if rank == p_root and verbose:
                            print('should have updated a maximum of', len(X_setmin_aseq_i_totry) - len(X_i), \
                                'i values in eles_lowest_i_failthresh:', \
                                'np.sum(eles_lowest_i_failthresh==i)', np.sum(eles_lowest_i_failthresh==i))
                            print('len(I2) is', len(I2), 'len(X_i) is', len(X_i), 'len(X) is', len(X), ' looking right', \
                                'len a_seq is', len(a_seq), 'checking for i in range',  I2[0], I2[-1] )

                    elif (len_X_i_trues < (1.0-2.0*eps)*len(X_setmin_aseq_i)):
                        # I_star found.
                        # Final element Did not pass threshold, so final i_star is one left in I2 (or -1 if we are currently at i=0)
                        i_notpass_idx = np.where(I2[0] == I2_copy)[0][0]
                        if i_notpass_idx >0:
                            i_star = I2_copy[i_notpass_idx-1]
                        else:
                            i_star = -1

                        if rank == p_root and verbose:
                            print('I2[0] did not pass', 'I2[0]=', I2[0], 'I2_copy[np.where(I2[0] == I2_copy)[0][0]]=',\
                                I2_copy[np.where(I2[0] == I2_copy)[0][0]], 'np.where(I2[0] == I2_copy)[0][0]=', \
                                np.where(I2[0] == I2_copy)[0][0], 'i_star=', i_star)
                    else:
                        # I_star found.
                        # Final element PASSED threshold, so final i_star is remaining element in I2
                        i_star = I2[0]

                    S_rounds.append([ele for ele in S])
                    time_rounds.append( MPI.Wtime() - p_start ) 
                    query_rounds.append(queries)   
                        

                # Add the new sequence a_i_star to S
                a_seq_to_add = a_seq[:(i_star+1)]
                S = list(np.sort(list(set().union(S, a_seq_to_add))))
                S_rounds[-1] = [ele for ele in S]
                time_rounds[-1] =  MPI.Wtime() - p_start 
                query_rounds[-1] = queries

                if rank == p_root and verbose:
                    print('i_star is', i_star, 'len a_seq is', len(a_seq), 'len(S)=', len(S))
                if len(S) == k:
                    break

                X = list(np.sort(list(set(X)-set(S))))

    #return S_star, I1_found
    comm.barrier()
    time = MPI.Wtime() - p_start
    if rank == p_root:
        print ('Parallel FAST:', objective.value(S), queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
        print('preprocess_add=', preprocess_add, 'lazy_binsearch=', lazy_binsearch, 'lazyouterX=', lazyouterX, 'debug_asserts=', debug_asserts,\
             'weight_sampling_eps=', weight_sampling_eps, 'sample_threshold=', sample_threshold, 'lazyskipbinsearch=', lazyskipbinsearch)

    # Print a warning if the speedups are turned off
    speedups_vals  = [preprocess_add,    lazy_binsearch,   lazyouterX,   sample_threshold,   lazyskipbinsearch]
    speedups_names = ['preprocess_add', 'lazy_binsearch', 'lazyouterX', 'sample_threshold', 'lazyskipbinsearch']
    if ((rank == p_root) and (not np.all(speedups_vals))):
        print('\n*NOTE*: the following optimizations are currently TURNED *OFF*. Turn them on (to True) in fn call to go faster!')
        print([s for idx, s in enumerate(speedups_names) if not speedups_vals[idx]])

    return objective.value(S), queries, time, S, S_rounds, time_rounds, query_rounds



def FAST_guessopt(objective, k, eps, preprocess_add=True, lazy_binsearch=True, \
                            lazyouterX=True, debug_asserts=False, weight_sampling_eps=1.0, \
                            sample_threshold=True, lazyskipbinsearch=True, allvals=None, verbose=False, \
                            stop_if_approx=True, eps_guarantee=0, seed=42):
    '''
    FAST Algorithm for Submodular Mazimization, VERSION FOR CASE WHERE OPT IS UNKNOWN (Breuer, Balkanski, Singer 2019)
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1

    OPTIONAL INPUTS:
    bool preprocess_add: turns on/off the 'Preprocessing  the  sequence' speedup
    bool lazy_binsearch: turns on/off lazy updates within the binary search for i^*
    bool lazyouterX: turns on/off lazy updates in outer while loop for X
    bool debug_asserts: turns on/off many debugging assert statements to debug parallel implementations of various functions
    float weight_sampling_eps: change from 1.0 to reweight different epsilons in algo separately. Increasing accelerates binseach for i^*
    bool sample_threshold: turns on/off sampling elements within the binary search for i^* (vs. trying them all)
    bool lazyskipbinsearch: turns on/off the speedup where if preprocessing is successful we can skip binary search for i^*
    list of floats allvals: if you provide a list of the value f(e) for e in the groundset (in order of groundset), this accelerates the algo
    bool verbose: turns on/off many print statements
    bool stop_if_approx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    int seed -- random seed to use when drawing samples

    OUTPUTS:
    float f(S) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list S -- the solution, where each element in the list is an element in the solution set.
    list of lists S_rounds -- each element is a list containing the solution set S at the corresponding round
    '''
    n = len(objective.groundset)
    time0 = datetime.now()

    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    proc_random_state = np.random.RandomState(seed)

    queries = 0
    queries_vec = [n, k, eps] #[]

    S_star = []
    f_S_star = 0
    #I1 = make_I(eps, k)

    if allvals is None:
        allvals = [objective.value([ele]) for ele in objective.groundset]
    queries += len(objective.groundset)
    queries_vec.append(len(objective.groundset))

    val_sum_topk = np.sum( np.sort(allvals)[-k:] )


    I1 = make_I_from_topksingletons(eps, k, np.max(allvals), val_sum_topk)
    len_I1_orig = len(I1)


    finished = False
    iter_outer_loop1 = 0


    if stop_if_approx:
        # Start at the highest value we want to try for OPT. 
        # This often works, then we are finished in a single iteration of this outer while-loop!
        midI1 = len(I1)-1 
    else:
        midI1 = len(I1) // 2

    while len(I1)>=1 and not finished:
        iter_outer_loop1+=1

        print ('Commencing I1 iter ', iter_outer_loop1, 'of', int( np.ceil( np.log2(len_I1_orig)+2 ) ), \
                    ', seconds elapsed =', (datetime.now()-time0).total_seconds())
        finished = len(I1)==1 
        #v = I1[midI1] * best_ele_val
        v = I1[midI1]# * best_ele_val

        f_S, queries_I2, time_I2, S_I2, _, _, _ = FAST_knowopt(objective, \
                                                                k, \
                                                                eps, \
                                                                v, \
                                                                preprocess_add, \
                                                                lazy_binsearch, \
                                                                lazyouterX, \
                                                                debug_asserts, \
                                                                weight_sampling_eps,\
                                                                sample_threshold, \
                                                                lazyskipbinsearch, \
                                                                allvals, \
                                                                OPT_guess_count=iter_outer_loop1, \
                                                                verbose=verbose, \
                                                                seed=seed)

        queries +=  queries_I2 
        queries_vec.append(queries_I2)


        if f_S > f_S_star:#objective.value(S_star):
            S_star = [ele for ele in S_I2]
            f_S_star = f_S

        #if f_S < (1.0-1.0/np.exp(1))*v:
        if f_S < (1.0-1.0/np.exp(1))*v:
            # keep lower half I1
            #print( '\n\nLOOKING LEFT IN I1')
            I1 = I1[:midI1]

        else:
            #print ('\n\nLOOKING RIGHT IN I1')
            I1 = I1[midI1:]
            #I1_found = True

        # Break early if we hit our theoretical guarantee. This speeds up the algorithm, 
        #but may possibly result in a lower value f(S) of the solution.

        if stop_if_approx:
            print('CHECKING APPROXIMATION GUARANTEE:', 'f_S_star', f_S_star, \
                '(1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk', (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk)

        if stop_if_approx and f_S_star >= (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk:
            print('APPROXIMATION GUARANTEE REACHED. STOPPING EARLY', 'f_S_star', f_S_star, \
                '(1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk', (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk)
            break

        # Update binary search location
        midI1 = len(I1) // 2


    time = (datetime.now() - time0).total_seconds()

    print(queries_vec)
    print('len queries_vec -3:', len(queries_vec)-3)
    return f_S_star, queries, time, S_star



def FAST_guessopt_parallel(objective, k, eps, comm, rank, size, preprocess_add=True, lazy_binsearch=True, \
                            lazyouterX=True, debug_asserts=False, weight_sampling_eps=1.0, \
                            sample_threshold=True, lazyskipbinsearch=True, allvals=None, verbose=False, \
                            stop_if_approx=True, eps_guarantee=0, p_root=0, seed=42):
    '''
    FAST Algorithm for Submodular Mazimization, VERSION FOR CASE WHERE OPT IS UNKNOWN (Breuer, Balkanski, Singer 2019)
    PARALLEL IMPLEMENTATION
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())

    OPTIONAL INPUTS:
    bool preprocess_add: turns on/off the 'Preprocessing  the  sequence' speedup
    bool lazy_binsearch: turns on/off lazy updates within the binary search for i^*
    bool lazyouterX: turns on/off lazy updates in outer while loop for X
    bool debug_asserts: turns on/off many debugging assert statements to debug parallel implementations of various functions
    float weight_sampling_eps: change from 1.0 to reweight different epsilons in algo separately. Increasing accelerates binseach for i^*
    bool sample_threshold: turns on/off sampling elements within the binary search for i^* (vs. trying them all)
    bool lazyskipbinsearch: turns on/off the speedup where if preprocessing is successful we can skip binary search for i^*
    list of floats allvals: if you provide a list of the value f(e) for e in the groundset (in order of groundset), this accelerates the algo
    bool verbose: turns on/off many print statements
    bool stop_if_approx: determines whether we exit as soon as we find OPT that reaches approx guarantee or keep searching for better solution
    int seed -- random seed to use when drawing samples

    OUTPUTS:
    float f(S) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list S -- the solution, where each element in the list is an element in the solution set.
    list of lists S_rounds -- each element is a list containing the solution set S at the corresponding round.
    '''    
    n = len(objective.groundset)

    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    proc_random_state = np.random.RandomState(seed)
    comm.barrier()
    p_start = MPI.Wtime()
    queries = 0
    queries_vec = [n, k, eps] #[]

    S_star = []
    S_rounds = [[]]
    query_rounds = [0]
    f_S_star = 0
    #I1 = make_I(eps, k)

    #best_ele_val, best_ele_idx = parallel_vals_faster(objective, [ele for ele in objective.groundset], comm, rank, size)
    if allvals is None:
        allvals = parallel_margvals_returnvals(objective, [], [ele for ele in objective.groundset], comm, rank, size)
    queries += len(objective.groundset)
    queries_vec.append(len(objective.groundset))

    val_sum_topk = np.sum( np.sort(allvals)[-k:] )


    I1 = make_I_from_topksingletons(eps, k, np.max(allvals), val_sum_topk)
    len_I1_orig = len(I1)


    finished = False
    iter_outer_loop1 = 0


    if stop_if_approx:
        # Start at the highest value we want to try for OPT. 
        # This often works, then we are finished in a single iteration of this outer while-loop!
        midI1 = len(I1)-1 
    else:
        midI1 = len(I1) // 2

    while len(I1)>=1 and not finished:
        iter_outer_loop1+=1

        if rank==p_root:
            print ('Commencing I1 iter ', iter_outer_loop1, 'of', int( np.ceil( np.log2(len_I1_orig)+2 ) ), \
                    ', seconds elapsed =', MPI.Wtime()-p_start)
        finished = len(I1)==1 
        #v = I1[midI1] * best_ele_val
        v = I1[midI1]# * best_ele_val
        #Tracking adaptivity
        f_S, queries_I2, time_I2, S_I2, sol_r, time_r, qry_r = FAST_knowopt_parallel( objective, \
                                                                        k, \
                                                                        eps, \
                                                                        v, \
                                                                        comm,\
                                                                        rank, \
                                                                        size, \
                                                                        preprocess_add, \
                                                                        lazy_binsearch, \
                                                                        lazyouterX, \
                                                                        debug_asserts, \
                                                                        weight_sampling_eps,\
                                                                        sample_threshold, \
                                                                        lazyskipbinsearch, \
                                                                        allvals, \
                                                                        OPT_guess_count=iter_outer_loop1, \
                                                                        verbose=verbose, \
                                                                        p_root=p_root, \
                                                                        seed=seed)

        queries +=  queries_I2 
        queries_vec.append(queries_I2)
        #Tracking adaptivity
        if(len(S_rounds)==1):
            S_rounds = sol_r
        else:
            for i in range(len(sol_r)):
                S_rounds.append(sol_r[i])
                query_rounds.append(qry_r[i])

        if f_S > f_S_star:#objective.value(S_star):
            S_star = [ele for ele in S_I2]
            f_S_star = f_S

        #if f_S < (1.0-1.0/np.exp(1))*v:
        if f_S < (1.0-1.0/np.exp(1))*v:
            # keep lower half I1
            #print( '\n\nLOOKING LEFT IN I1')
            I1 = I1[:midI1]

        else:
            #print ('\n\nLOOKING RIGHT IN I1')
            I1 = I1[midI1:]
            #I1_found = True

        # Break early if we hit our theoretical guarantee. This speeds up the algorithm, 
        #but may possibly result in a lower value f(S) of the solution.

        if stop_if_approx and rank==p_root:
            print('CHECKING APPROXIMATION GUARANTEE:', 'f_S_star', f_S_star, \
                '(1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk', (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk)

        if stop_if_approx and f_S_star >= (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk:
            if rank==p_root:
                print('APPROXIMATION GUARANTEE REACHED. STOPPING EARLY', 'f_S_star', f_S_star, \
                    '(1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk', (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk)
            break

        # Update binary search location
        midI1 = len(I1) // 2



    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)

    if rank == p_root:
        print(queries_vec)
        print('len queries_vec -3:', len(queries_vec)-3)


    # Print a warning if the speedups are turned off
    speedups_vals  = [preprocess_add,    lazy_binsearch,   lazyouterX,   sample_threshold,   lazyskipbinsearch,   stop_if_approx]
    speedups_names = ['preprocess_add', 'lazy_binsearch', 'lazyouterX', 'sample_threshold', 'lazyskipbinsearch', 'stop_if_approx']
    if ((rank == p_root) and (not np.all(speedups_vals))):
        print('\n*NOTE*: the following optimizations are currently TURNED *OFF*. Turn them on (to True) in fn call to go faster!')
        print([s for idx, s in enumerate(speedups_names) if not speedups_vals[idx]])

    #Tracking adaptivity
    return f_S_star, queries, time, S_star, S_rounds, query_rounds



def make_I_original(eps, k):
    last_ele = 0
    I = []
    ii = 0
    while last_ele < k:
        last_ele = 1.0/(1.0-eps)**ii
        I.append(last_ele)
        ii+=1
    I[-1] = k 
    print('ASK ERIC ABOUT MAKING LAST ELEMENT EXACTLY k. without this line its a bit MORE than k')
    print('ALSO ASK about making the first element 0, since we never actually use that element in practice I think maybe.')
    return I  #print last_ele



def make_I2_0idx(eps, k):
    last_ele = 0
    I = [0]
    ii = 0
    #while last_ele < (k-1):
    while last_ele < (k):
        '^^k-1??'
        last_ele = 1.0/(1.0-eps)**ii
        I.append(last_ele)
        ii+=1
    #I[-1] = k -1
    I[-1] = k 
    '^^k-1??'
    #print 'ASK ERIC ABOUT MAKING LAST ELEMENT EXACTLY k. without this line its a bit MORE than k'
    return np.unique(np.floor(I)).astype(np.int64)    #print last_ele



def make_I_from_topksingletons(eps, k, val_best_singleton, val_sum_topk):
    # val_best_singleton is the value f(v*) where v* is the best-valued singleton elements.
    # val_sum_topk is the sum of values of the the top k best-valued singleton elements.
    last_ele = 0
    I = [val_best_singleton]
    ii = 0
    while last_ele < val_sum_topk:
        last_ele = 1.0/(1.0-eps)**ii
        I.append(last_ele * val_best_singleton)
        ii += 1
    I[-1] = val_sum_topk 
    return I  #print last_ele



def F_x_ofxprime(objective, x, xprime, numsamples):
    assert(len(x)==len(xprime)==len(objective.groundset))
    samples = np.random.random((numsamples, len(x)))
    # Draw binary vectors of length n where each element is true with probability x[j]
    samples_x = [samples[s] < x for s in range(numsamples)]
    # Draw binary vectors of length n where each element is true with probability xprime[j]
    samples_xprime = [samples[s] < xprime for s in range(numsamples)]

    out_vec =  [objective.value( list(np.where(np.maximum(samples_x[ss], samples_xprime[ss]))[0] )) \
                    - objective.value(list(np.where(samples_x[ss])[0])) for ss in range(numsamples) ]
    #print (out)#, 'x', x, 'xprime', xprime)
    return np.mean(out_vec)#, np.sum(out_vec)



def F_x_ofxprime_parallel(objective, x, xprime, numsamples, randomstate_local, comm, rank, size, p_root=0):
    comm.barrier()

    assert(len(x)==len(xprime)==len(objective.groundset))
    numsamples_local = int(np.ceil(np.float(numsamples)/size))

    samples = randomstate_local.rand(numsamples_local, len(x))
    # Draw binary vectors of length n where each element is true with probability x[j]
    samples_x = [samples[s] < x for s in range(numsamples_local)]
    # Draw binary vectors of length n where each element is true with probability xprime[j]
    samples_xprime = [samples[s] < xprime for s in range(numsamples_local)]

    myinput = list(np.where(np.maximum(samples_x[0], samples_xprime[0]))) 
    # if rank == p_root:
    #     print(myinput, 'myinput') #DELETE
    comm.barrier()

    local_sum = np.sum( [objective.value( list(np.where(np.maximum(samples_x[ss], samples_xprime[ss]))[0]) ) \
                     - objective.value(list(np.where(samples_x[ss])[0])) for ss in range(numsamples_local) ] )
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_mean = global_sum / np.float(size*numsamples_local)
    #print('global mean', global_mean, 'rank', rank, 'local_mean', local_mean) # rank0 proc should be smaller local mean
    #print('rank', rank, 'local mean', local_sum/np.float(numsamples_local), 'global mean', global_mean)
    return global_mean#, global_sum




def choosed_approx_fig2_stopwhencond(objective, Q, S, eps, t, k, lam, numsamples, d_vec_to_try):
    ''' make d (scalar float) maximal but satisfying the 3 constraints. '''
    'in choosed_approx_fig2 ASK ERIC what to do about d=0, and also if we even try d=0. lets not for now'
    assert (not bool(set(Q) & set(S))) # set S and set Q should be disjoint or we will have issues

    # Need to make list of elements Q into binary vector
    n = len(objective.groundset)
    q = np.zeros(n)
    q[Q] = 1.0

    d = 0#. stepsize_d_opt, but then need to q_plus_ds[S] += d

    choosed_queries = 0
    broke_for_failed_condition = False
    all_d_too_large = False


    for ii in range(len(d_vec_to_try)):

        d = d_vec_to_try[ii]

        q_plus_ds = np.copy(q)
        q_plus_ds[S] = d

        cond1 = ( F_x_ofxprime(objective, q, q_plus_ds, numsamples) >= (1.0-eps)**2 * lam*d*np.float(len(S))/np.float(k)  )
        cond2 = (t + d*np.float(len(S))) <= (1.0-2.0*eps)*np.float(k)
        choosed_queries += numsamples

        if not (cond1 and cond2):
            print( 'breaking for whichever is false: \
                    cond1:', cond1, 'or cond2:', cond2, 'len(S)=', len(S), 'len(Q)=', len(Q) )
            #print 'cond1:', F_x_ofxprime(objective, q, q_plus_ds, numsamples), \
            #    (1.0-eps)**2 * lam*d*np.float(len(S))/np.float(k) 
            broke_for_failed_condition = True
            break

    if (broke_for_failed_condition and (ii >=1)):
        d = d_vec_to_try[ii-1]

    elif broke_for_failed_condition:
        print( 'First d tried is too large and fails conditions')
        all_d_too_large = True

    print( 'd returning', d)
    return d, choosed_queries, all_d_too_large, cond1, cond2




def choosed_approx_fig2_stopwhencond_parallel(objective, Q, S, eps, t, k, lam, numsamples, d_vec_to_try, \
                                                randomstate_local, comm, rank, size, p_root=0):
    ''' make d (scalar float) maximal but satisfying the 3 constraints. '''
    'in choosed_approx_fig2 ASK ERIC what to do about d=0, and also if we even try d=0. lets not for now'
    assert (not bool(set(Q) & set(S))) # set S and set Q should be disjoint or we will have issues

    # Need to make list of elements Q into binary vector
    n = len(objective.groundset)
    q = np.zeros(n)
    q[Q] = 1.0

    d = 0#. stepsize_d_opt, but then need to q_plus_ds[S] += d

    choosed_queries = 0
    broke_for_failed_condition = False
    all_d_too_large = False


    for ii in range(len(d_vec_to_try)):

        d = d_vec_to_try[ii]

        q_plus_ds = np.copy(q)
        q_plus_ds[S] = d

        comm.barrier()
        #cond1_allLocal_val = F_x_ofxprime(objective, q, q_plus_ds, numsamples)
        #cond1_allLocal = ( F_x_ofxprime(objective, q, q_plus_ds, numsamples) >= (1.0-eps)**2 * lam*d*np.float(len(S))/np.float(k)  )
        
        cond1_val = F_x_ofxprime_parallel(objective, q, q_plus_ds, numsamples, randomstate_local, comm, rank, size, p_root=0)

        cond1 = cond1_val >= ( (1.0-eps)**2 * lam*d*np.float(len(S))/np.float(k)  )
        cond2 = (t + d*np.float(len(S))) <= (1.0-2.0*eps)*np.float(k)
        choosed_queries += numsamples

        #print('is f_x_of_xprime_OK?', 'rank', rank, cond1, F_x_ofxprime(objective, q, q_plus_ds, numsamples), \
        #        F_x_ofxprime_parallel(objective, q, q_plus_ds, numsamples, randomstate_local, comm, rank, size, p_root=0))
        #print('cond1_allLocal_val', cond1_allLocal_val, 'cond1_val', cond1_val)
        #print('\n')

        if not (cond1 and cond2):
            if rank == p_root:
                print( 'breaking for whichever is false: cond1:', cond1, 'or cond2:', cond2, 'len(S)=', len(S), 'len(Q)=', len(Q) )
            #print 'cond1:', F_x_ofxprime(objective, q, q_plus_ds, numsamples), (1.0-eps)**2 * lam*d*np.float(len(S))/np.float(k) 
            broke_for_failed_condition = True
            break

    if (broke_for_failed_condition and (ii >=1)):
        d = d_vec_to_try[ii-1]

    elif broke_for_failed_condition:
        if rank == p_root:
            print( 'First d tried is too large and fails conditions')
        all_d_too_large = True

    if rank == p_root:
        print( 'd returning', d)
    return d, choosed_queries, all_d_too_large, cond1, cond2




def make_opts_randparagreedy(objective, eps, k):
    allvals = [objective.value([ele]) for ele in objective.groundset]
    val_sum_topk = np.sum( np.sort(allvals)[-k:] )
    return [val_sum_topk] #print last_ele



def make_opts_randparagreedy_parallel(objective, eps, k, comm, rank, size):
    allvals = parallel_margvals_returnvals(objective, [], [ele for ele in objective.groundset], comm, rank, size)
    val_sum_topk = np.sum( np.sort(allvals)[-k:] )
    return [val_sum_topk] #print last_ele



# def make_deltas_randparagreedy(objective, eps, k, coefficient=1.0):
#     last_ele = eps/np.float(len(objective.groundset))
#     delta_randparagreedy = []
#     #ii = 0
#     while last_ele < 1.0:
#         last_ele = last_ele * (1.0/(1.0-coefficient*eps))
#         delta_randparagreedy.append(last_ele)
#         #ii+=1
#     delta_randparagreedy[-1] = 1.0
#     print(delta_randparagreedy)

#     return np.array(delta_randparagreedy)



# def make_deltas_basic_randparagreedy(objective, eps, k, num_steps=100):
#     return np.linspace(1./num_steps, 1, num_steps-1)



def make_deltas_randparagreedy(objective, eps, k, num_steps=500):
    return np.linspace(1./num_steps, 1, num_steps-1)



def randparagreedy(objective, k, eps, OPT, numsamples, d_vec_to_try, seed=42):
    '''
    Chekuri et al. randomized-parallel-greedy for cardinality constraints. Algorithm from Figure 2 p. 12 of
    "Submodular Function Maximization in Parallel via the Multilinear Relaxation"
    *** NOTE *** See below for a different implementation used when OPT is not known and several OPTS must be tried to 
    get the theoretical performance guarantee.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    float OPT -- the value of the optimal solution S*
    int numsamples -- the sample complexity
    d_vec_to_try -- a list of floats of various deltas to try in the algorithm
    int seed -- random seed
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list Q -- the solution set of elements
    list of lists Q_rounds -- each element is a list containing the solution set Q at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in Q_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    '''
    check_inputs(objective, k)

    # Storage
    queries = 0
    time0 = datetime.now()

    # Initialize
    n = len(objective.groundset)
    Q = []
    t = 0.0
    lam = OPT
    randstate = np.random.RandomState(seed)

    Q_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]

    iter_outerloop = 0

    while ( (t <= ((1.0-2.0*eps)*np.float(k) - np.min(d_vec_to_try)) ) and ( lam >= (OPT / np.exp(1)) ) and len(Q)<k ):
        print ('f(S)=', objective.value(Q), 'randparagreedy outerloop iter=', iter_outerloop, \
            't=', t, 'lam=', lam, '(OPT / np.exp(1))=', (OPT / np.exp(1)))

        # S is a set of high marg value elements
        elements_not_in_sol = list(set(objective.groundset) - set(Q))
        S = [ ele for ele in elements_not_in_sol if (objective.marginalval([ele], Q) >= (1-eps)*lam/np.float(k) ) ]
        queries+= len(elements_not_in_sol) + 1 # +1 is because Q changed by a random set so we need to eval it again

        # if not S:
        #     print '\n\n\nS IS EMPTY SO RETURNING. 
        'CHECK THIS AS ITS NOT IN ORIG ALGO WRITEUP'
        #     return Q, queries

        print( 'len(S)', len(S))
        iter_innerloop = 0
        while ( np.any(S) and  (t <= (np.float(k)*(1.0 - 2.0*eps) - np.min(d_vec_to_try)) ) ): 
            # NOTE the -stepsize there bc otherwise we endlessly loop where t==0.9999 and cant go higher

            'DIDNT DO QUERIES FOR THIS PART YET'
            'ASK ERIC about CHOOSEd brute force up to break or brute force all in vec?'

            d, choosed_queries, all_d_too_large, sum_cond1, sum_cond2 = \
                choosed_approx_fig2_stopwhencond(objective, Q, S, eps, t, k, lam, numsamples, d_vec_to_try)
            queries += choosed_queries


            if all_d_too_large:
                print( 'PROGRESS AT AN END AS d=0. im doing a KEEP GOING WITH 0 and go to new OUTER LOOP (draw new S) but ASK ERIC')
                print( 'sum_cond1', sum_cond1, 'sum_cond2', sum_cond2, 'outer_loop_t_cond_similarto_cond2', \
                    t <= ((1.0-2.0*eps)*np.float(k)), 't', t, '<=((1.0-2.0*eps)*np.float(k))', ((1.0-2.0*eps)*np.float(k)))
                break

            print ('adding each element in S with probability d=', d)
            # Draw a random set. Basically, we're going to draw each of the elements in S with low probability
            S_nvec = np.zeros(n)
            S_nvec[S] = 1.0


            # R_vec = (np.random.random(size=n) < d*S_nvec)
            R_vec = (randstate.rand(n) < d*S_nvec)
            R = [idx for idx, ele in enumerate(R_vec) if ele]

            # Add the True draws to solution Q
            #print ('Q size before R', len(Q))
            Q = list(set().union(Q, R))
            #print( 'Q size after R', len(Q))

            Q_rounds.append([ele for ele in Q])
            time_rounds.append( (datetime.now() - time0).total_seconds() )
            query_rounds.append(queries)

            # Update the current expected cost
            t = t + d*len(S)

            # Remove elements from S if we added them to solution
            #print 'ASK ERIC about line 2.B.iv does it just mean S = S-R'
            elements_not_in_sol = list(set(objective.groundset) - set(Q))
            S = [ ele for ele in elements_not_in_sol if (objective.marginalval([ele], Q) >= (1-eps)*lam/np.float(k) ) ]
            queries += len(elements_not_in_sol) + 1 #+1 is because Q changed by a random set so we need to eval it again
            iter_innerloop +=1

        lam = (1.0-eps)*lam
        iter_outerloop +=1

    Q = Q[:(k+1)]
    Q_value = objective.value(Q)
    time = (datetime.now() - time0).total_seconds()
    Q_rounds[-1] = [ele for ele in Q]
    time_rounds[-1] = time
    query_rounds[-1] = queries
    return Q_value, queries, time, Q, Q_rounds, time_rounds, query_rounds




def randparagreedy_parallel(objective, k, eps, OPT, numsamples, d_vec_to_try, comm, rank, size, p_root=0, seed=42):
    '''
    PARALLELIZED Chandra's parallel-greedy for cardinality constraints. Algorithm from Figure 2 p. 12.
    *** NOTE *** See below for a different implementation used when OPT is not known and several OPTS must be tried to 
    get the theoretical performance guarantee.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    float OPT -- the value of the optimal solution S*
    int numsamples -- the sample complexity
    d_vec_to_try -- a list of floats of various deltas to try in the algorithm
    float eps -- the error tolerance between 0 and 1
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    
    OPTIONAL INPUTS:
    int p_root -- the rank of the processor we want to use as root/master
    seed -- a seed to seed the random set generator. Setting this causes the function to replicate the same result.
    
    OUTPUTS:
    float f(Q) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list Q -- the solution set of elements
    list of lists Q_rounds -- each element is a list containing the solution set Q at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in Q_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.

    '''
    check_inputs(objective, k)

    # Each processor gets UNIQUE random state such that they each draw different random samples
    randomstate_local = np.random.RandomState(rank+seed)

    # Storage
    queries = 0
    comm.barrier()
    p_start = MPI.Wtime()

    # Initialize
    n = len(objective.groundset)
    Q = []
    t = 0.0
    lam = OPT
    iter_outerloop = 0

    Q_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]


    while ( (t <= ((1.0-2.0*eps)*np.float(k) - np.min(d_vec_to_try)) ) and ( lam >= (OPT / np.exp(1)) ) and len(Q)<k ):
        if rank == p_root:
            print ('f(S)=', objective.value(Q), 'randparagreedy_parallel outerloop iter=', iter_outerloop, \
                't=', t, 'lam=', lam, '(OPT / np.exp(1))=', (OPT / np.exp(1)))

        # S is a set of high marg value elements
        elements_not_in_sol = list(set(objective.groundset) - set(Q))
        threshold = (1-eps)*lam/np.float(k)
        comm.barrier()

        S = parallel_margvals_returneles_ifthresh(objective, Q, elements_not_in_sol, threshold, comm, rank, size)
        comm.barrier()



        queries += len(elements_not_in_sol) + 1 # +1 is because Q changed by a random set so we need to eval it again
        Q_rounds.append([ele for ele in Q])
        time_rounds.append( MPI.Wtime() - p_start )
        query_rounds.append(queries)

        
        iter_innerloop = 0
        while ( np.any(S) and  (t <= (np.float(k)*(1.0 - 2.0*eps) - np.min(d_vec_to_try)) ) ): 
            # NOTE the -stepsize there bc otherwise we endlessly loop where t==0.9999 and cant go higher

            #print ('DIDNT DO QUERIES FOR THIS PART YET')
            #print ('ASK ERIC about CHOOSEd brute force up to break or brute force all in vec?')
            # dlocal, choosed_queriesloc, all_d_too_largeloc, sum_cond1loc, sum_cond2loc = \
            # choosed_approx_fig2_stopwhencond(objective, Q, S, eps, t, k, lam, numsamples, d_vec_to_try)

            #print('\n\n\n\n', 'now parallel version of choosedapprox')
            comm.barrier()
            d, choosed_queries, all_d_too_large, sum_cond1, sum_cond2 = \
                choosed_approx_fig2_stopwhencond_parallel(  objective, \
                                                            Q, \
                                                            S, \
                                                            eps, \
                                                            t, \
                                                            k, \
                                                            lam, \
                                                            numsamples, \
                                                            d_vec_to_try, \
                                                            randomstate_local, \
                                                            comm, \
                                                            rank, \
                                                            size, \
                                                            p_root=0)
            queries += choosed_queries

            Q_rounds.append([ele for ele in Q])
            time_rounds.append( MPI.Wtime() - p_start )
            query_rounds.append(queries)

            #if rank == p_root:
            #print('\n')
            # print('rank', rank, 'd', d, 'dlocal', dlocal, 'all_d_too_large', all_d_too_large, \
            #       'all_d_too_largeloc', all_d_too_largeloc, 'sum_cond1', sum_cond1, 'sum_cond1loc', sum_cond1loc,\
            #        'sum_cond2', sum_cond2, 'sum_cond2loc', sum_cond2loc)

            # print('rank', rank, 'd', d, 'all_d_too_large', all_d_too_large, \
            #       'sum_cond1', sum_cond1,\
            #        'sum_cond2', sum_cond2, )

            # if rank == p_root:
            #     print( 'd_orig', d)

            if all_d_too_large:
                if rank == p_root:
                    print( 'sum_cond1', sum_cond1, 'sum_cond2', sum_cond2, 'outer_loop_t_cond_similarto_cond2',\
                     t <= ((1.0-2.0*eps)*np.float(k)), 't', t, '<=((1.0-2.0*eps)*np.float(k))', ((1.0-2.0*eps)*np.float(k)))
                break

            # if rank == p_root:
            #     print ('adding each element in S with probability d=', d)

            # Draw a random set. Basically, we're going to draw each of the elements in S with low probability
            # Root processor draws new set R of high val elements in S to add to solution
            if rank == p_root:
                S_nvec = np.zeros(n)
                S_nvec[S] = 1.0
                R_vec = (randomstate_local.rand(n) < d*S_nvec)
                R = [idx for idx, ele in enumerate(R_vec) if ele]

                # Add the True draws to solution Q

                # if rank == p_root:
                #     print ('Q size before R', len(Q))
                Q = list(set().union(Q, R))
                # if rank == p_root:
                #     print( 'Q size after R', len(Q))

            Q = comm.bcast(Q, root=0)

            # Update the current expected cost
            t = t + d*len(S)

            # Remove elements from S if we added them to solution
            #print 'ASK ERIC about line 2.B.iv does it just mean S = S-R'
            comm.barrier()
            elements_not_in_sol = list(set(objective.groundset) - set(Q))
            threshold = (1-eps)*lam/np.float(k)

            S = parallel_margvals_returneles_ifthresh(objective, Q, elements_not_in_sol, threshold, comm, rank, size)
            queries += len(elements_not_in_sol) + 1 #+1 is because Q changed by a random set so we need to eval it again
            iter_innerloop +=1

            Q_rounds.append([ele for ele in Q])
            time_rounds.append( MPI.Wtime() - p_start )
            query_rounds.append(queries)

        lam = (1.0-eps)*lam
        comm.barrier()
        iter_outerloop +=1

    Q = Q[:(k+1)]
    Q_rounds[-1] = [ele for ele in Q]
    Q_value = objective.value(Q)

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    time_rounds[-1] = time
    query_rounds[-1] = queries
    return Q_value, queries, time, Q, Q_rounds, time_rounds, query_rounds




def randparagreedyOPTS(objective, k, eps, numsamples):
    '''Wrapper to generate the necessary OPTS and deltas for the theoretical guarantee and run Chandra Fig2 over them'''
    time0 = datetime.now()
    queries = 0

    opts_randparagreedy = make_opts_randparagreedy(objective, eps, k)
    queries += len(objective.groundset)
    #opts_randparagreedy = [15770.]

    n = len(objective.groundset)
    deltas_randparagreedy = make_deltas_randparagreedy(objective, eps, k, num_steps=len(objective.groundset))
    #deltas_randparagreedy = np.arange(0.01, 1.0, 0.01)

    f_Q = 0
    Q_star = []
    for oo, OPT in enumerate(opts_randparagreedy):
        print('randparagreedyOPTS main loop ', oo, 'of', len(opts_randparagreedy))
        f_Q_given_OPT, queries_OPT, _, Q_given_OPT, _, _, _  = randparagreedy(objective, k, eps, OPT, numsamples, deltas_randparagreedy)
        #f_Q = np.max([f_Q, f_Q_given_OPT])
        if f_Q < f_Q_given_OPT:
            f_Q = f_Q_given_OPT
            Q_star = [ele for ele in Q_given_OPT]
        queries += queries_OPT

    return f_Q, queries, (datetime.now()-time0).total_seconds(), Q_star




def randparagreedyOPTS_parallel(objective, k, eps, numsamples, comm, rank, size, p_root=0, seed=42):
    '''PARALLEL Wrapper to generate the necessary OPTS and deltas for the theoretical guarantee and run randparagreedy over them'''
    comm.barrier()
    p_start = MPI.Wtime()
    queries = 0

    opts_randparagreedy = make_opts_randparagreedy_parallel(objective, eps, k, comm, rank, size)

    queries += len(objective.groundset)

    n = len(objective.groundset)
    deltas_randparagreedy = make_deltas_randparagreedy(objective, eps, k, num_steps=len(objective.groundset))

    f_Q = 0
    Q_star = []
    for oo, OPT in enumerate(opts_randparagreedy):
        if rank == p_root:
            print('randparagreedyOPTS_parallel main loop ', oo, 'of', len(opts_randparagreedy))

        comm.barrier()
        f_Q_given_OPT, queries_OPT, _, Q_given_OPT, _, _, _ = randparagreedy_parallel( objective, \
                                                                                    k, \
                                                                                    eps, \
                                                                                    OPT, \
                                                                                    numsamples, \
                                                                                    deltas_randparagreedy, \
                                                                                    comm, \
                                                                                    rank, \
                                                                                    size, \
                                                                                    p_root,\
                                                                                    seed)
        #f_Q = np.max([f_Q, f_Q_given_OPT])
        #queries += queries_OPT + 1
        if f_Q < f_Q_given_OPT:
            f_Q = f_Q_given_OPT
            Q_star = [ele for ele in Q_given_OPT]
        queries += queries_OPT

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)
    return f_Q, queries, time, Q_star




def sampleX_seeded_parallel(X, k, rank, proc_random_states):
    if len(X) <= k:
        return X
    return list(proc_random_states[rank].choice(X, k, replace=False))




def sampleX_randstate(X, k, randstate):
    if len(X) <= k:
        return X
    return list(randstate.choice(X, k, replace=False))




def sampleX_wRepl_randstate(X, k, randstate):
    if len(X) <= k:
        return X
    return list(randstate.choice(X, k, replace=False))




def sampleX(X,k):
    if len(X) <= k:
        return X
    return list(np.random.choice(X, k, replace=False))




def sampleX_wRepl(X,k):
    if len(X) <= k:
        return X
    return list(np.random.choice(X, k, replace=True))




def estimate_margvalset_parallel(objective, X, S, numsamples_m, k, r, randomstate_local, comm, rank, size):
    # if size == 1:
    #     numsamples_m_local = numsamples_m
    # elif rank == p_root:
    #     numsamples_m_local = np.max((divmod(numsamples_m, size)[1], 1)) #can't be 0 or nan induced
    # else:
    #     numsamples_m_local = divmod(numsamples_m, size)[0]
    numsamples_m_local = int(np.ceil(np.float(numsamples_m)/size))

    samples_local = [ sampleX_wRepl_randstate(X, int( np.ceil(np.float(k)/r)), \
                        randomstate_local) for ss in range(numsamples_m_local) ]

    # Each processor computes marg value for its subset of samples
    value_randset_local = np.sum( [ objective.marginalval( list(set(ss)-set(S)), S ) for ss in samples_local ] )

    # Processors trade values and compute global mean
    sum_randset = comm.allreduce(value_randset_local, op=MPI.SUM)

    # if ((size>1) and (divmod(numsamples_m, size)[1] == 0)):
    #     value_randset = sum_randset/np.float(numsamples_m+1) #because we added a sample so root proc wouldnt have no samples
    # else:
    #     value_randset = sum_randset/np.float(numsamples_m)
    value_randset = sum_randset/np.float(numsamples_m_local*size)

    return value_randset 




def estimate_margvalset(objective, X, S, numsamples_m, k, r):
    #print 'SAMPLING WITH OR OUT REPLACEMENT estimate_margvalset ASK ERIC'
    #assert(len(X) >= np.ceil(np.float(k)/r))
    val = np.mean( [ objective.marginalval( list(set(sampleX_wRepl(X, int( np.ceil(np.float(k)/r)) ))-set(S)), S ) \
                        for ss in range(numsamples_m) ] )
    #queries = 2*numsamples_m
    return val#, queries




def estimatemargval_returnelesifthresh_parallel(objective, X, SuT, numsamples_m, k, r, ele_thresh, 
                                                randomstate_local, comm, rank, size, p_root=0):
    #print 'SAMPLING WITH OR OUT REPLACEMENT estimatemargval_returnelesifthresh ASK ERIC'
    queries_elefilter_local = 0
    X_noST = np.sort(list(set(X) - set(SuT)))
    #assert(len(X_noS) >= np.ceil(np.float(k)/r))
    #samplesR = [ sampleX_wRepl(X_noST, int( np.ceil(np.float(k)/r)) ) for ss in range(numsamples_m) ]
    # On root processor, draw numsamples new sample sets of remaining elements
    # that we might want to include in solution, each of size k/r. 
    if rank == p_root:
        samplesR = [ sampleX_wRepl_randstate(X, int( np.ceil(np.float(k)/r)), randomstate_local) \
                        for ss in range(numsamples_m) ]
    else:
        samplesR = None

    # Broadcast samples from root processor to p processors
    comm.barrier()
    #samplesR = comm.scatter(np.array_split(samplesR, size), root=0)
    samplesR = comm.bcast(samplesR, root=0)
    #print('rank', rank)
    comm.barrier()

    X_noST_local = np.array_split(X_noST, size)[rank]
    eles_above_thresh_local = []

    for ele in X_noST_local:
        #v_s_X_a = np.mean( [ objective.marginalval( list(set().union(SuT, Ri, [ele])), \
        #               list(set().union(SuT, Ri) - set([ele])) ) for Ri in samplesR ] )
        v_s_X_a = np.mean( [ objective.marginalval( [ele], list(set().union(SuT, Ri) - set([ele])) ) for Ri in samplesR ] )
        #queries_elefilter += 2*samplesR
        #print 'v_s_X_a', v_s_X_a, 'ele_thresh', ele_thresh
        if v_s_X_a >= ele_thresh:
            eles_above_thresh_local.append(ele)

    # Processors trade values so all get all.
    eles_above_thresh_nested = comm.allgather(eles_above_thresh_local)
    #print(eles_above_thresh_nested)
    return [ele for sublist in eles_above_thresh_nested for ele in sublist]# flattens the list




def estimatemargval_returnelesifthresh(objective, X, SuT, numsamples_m, k, r, ele_thresh):
    queries_elefilter = 0
    X_noST = list(set(X) - set(SuT))
    #assert(len(X_noS) >= np.ceil(np.float(k)/r))
    samplesR = [ sampleX_wRepl(X_noST, int( np.ceil(np.float(k)/r)) ) for ss in range(numsamples_m) ]
    eles_above_thresh = []

    for ele in X_noST:
        #v_s_X_a = np.mean( [ objective.marginalval( list(set().union(SuT, Ri, [ele])), \
        #list(set().union(SuT, Ri) - set([ele])) ) for Ri in samplesR ] )
        v_s_X_a = np.mean( [ objective.marginalval( [ele], list(set().union(SuT, Ri) - set([ele])) ) for Ri in samplesR ] )

        #queries_elefilter += 2*numsamples_m
        #print 'v_s_X_a', v_s_X_a, 'ele_thresh', ele_thresh
        if v_s_X_a >= ele_thresh:
            eles_above_thresh.append(ele)

    return eles_above_thresh#, queries_elefilter




def amortizedfiltering_parallel(objective, k, eps, OPT, numsamples_m, comm, rank, size, p_root=0, seed=42):
    '''
    Balkanski, Rubinstein, and Singer Amortized Filtering algorithm, SODA 2019
    *** NOTE *** See below for a different implementation used when OPT is not known and several OPTS must be tried to 
    get the theoretical performance guarantee.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    float OPT -- the value of the optimal solution S*
    int numsamples -- the sample complexity
    int r -- number of rounds
    comm -- the MPI4py Comm (MPI.COMM_WORLD)
    int rank -- the processor's rank (comm.Get_rank())
    int size -- the number of processors (comm.Get_size())
    
    OPTIONAL INPUTS:
    int p_root -- the rank of the processor we want to use as root/master
    int seed -- a seed to seed the random set generator. Setting this causes the function to replicate the same result.
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list S -- the list of elements in the solution
    list of lists S_rounds -- each element is a list containing the solution set S at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in S_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    '''
    check_inputs(objective, k)
    comm.barrier()
    p_start = MPI.Wtime()

    # All processors get UNIQUE random state such that they draw different samples!
    randomstate_local = np.random.RandomState(rank+seed)

    queries = 0
    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    r = int( np.ceil(20.0/eps) )

    no_more_ele_in_X_iters = 0

    for ii in range(r):

        X = list(set(objective.groundset) - set(S))
        T = []

        inner_loop_iter = 0
        assert (not bool(set(S) & set(T))) # set S and set T should be disjoint or we will have issues
        assert(len(S)==len(np.unique(S)))
        assert(len(T)==len(np.unique(T)))


        SuT = list(set().union(S, T))
        f_of_SuT = objective.value(SuT)
        f_of_S = objective.value(S)
        queries += 2

        S_rounds.append([ele for ele in list(set().union(S, T))])
        time_rounds.append(MPI.Wtime() - p_start)
        query_rounds.append(queries)


        if rank == p_root and (ii%25 == 0):
            print ('ii', ii, 'of', r, 'len(S)=', len(S), 'k', k, 'f_of_SuT', f_of_SuT)
        #print ('SAMPLING WITH OR OUT REPLACEMENT estimatemargval_returnelesifthresh ASK ERIC')



        # if no_more_ele_in_X_iters > maxiters_noX:
        #     if rank == p_root:
        #         print( 'Ran out of ground set elements that meet threshold after', maxiters_noX, 'iterations. Returning.')
        #     break

        #print (' FOR LOOP ii', ii, 'f_of_SuT', f_of_SuT, '< (eps/20.0)*(OPT - f_of_S) ',  \
        #    (eps/20.0)*(OPT - f_of_S) , 'len(SuT)', len(SuT), '<k', k)
        # while ( (no_more_ele_in_X_iters <= maxiters_noX) and \
        #    ((f_of_SuT - f_of_S) < (eps/20.0)*(OPT - f_of_S) ) and (len(SuT) < k) ):

        #     # If we run out of ground set elements that meet the threshold and \
        #     multiple tries (different samples) doesn't help, then break
        #     if len(X)==0:
        #         no_more_ele_in_X_iters += 1
        #     else:
        #         no_more_ele_in_X_iters = 0
        #  #print (' FOR LOOP ii', ii, 'f_of_SuT', f_of_SuT, '< (eps/20.0)*(OPT - f_of_S) ',  \
        # (eps/20.0)*(OPT - f_of_S) , 'len(SuT)', len(SuT), '<k', k)
        while ( len(X) and ((f_of_SuT - f_of_S) < (eps/20.0)*(OPT - f_of_S) ) and (len(SuT) < k) ):

            # If we run out of ground set elements that meet the threshold and multiple tries 
            # (i.e. different samples) doesn't help, then break
            if len(X)==0:
                no_more_ele_in_X_iters += 1
            else:
                no_more_ele_in_X_iters = 0               
            #print ('\n\n\n ENTERED OUTER WHILE LOOP')
            #print ('estimating val randset outer while loop')
            value_randset = estimate_margvalset_parallel(objective, X, SuT, numsamples_m, k, r, randomstate_local, comm, rank, size)
            queries += 2*numsamples_m

            S_rounds.append([ele for ele in list(set().union(S, T))])
            time_rounds.append(MPI.Wtime() - p_start)
            query_rounds.append(queries)

            ele_thresh = (1.0 + eps/2.0)*(1.0-eps)*(OPT-f_of_SuT)/np.float(k)

            while len(X) and (value_randset < (1.0 - eps) * (OPT - f_of_SuT)/np.float(r)):
                # if rank == p_root:
                #     print ('ii', ii, 'of', r, 'len(X)=', len(X), 'f_of_SuT', f_of_SuT, \
                #       'value_randset', value_randset, '<',  (1.0 - eps) * (OPT - f_of_SuT)/np.float(r))
                # print ('estimating X')

                X = estimatemargval_returnelesifthresh_parallel(\
                        objective, list(set(X)-set(SuT)), SuT, numsamples_m, k, r, \
                        ele_thresh, randomstate_local, comm, rank, size, p_root=0)
                # if rank == p_root:
                #     print('len(X) filtered', len(X))
                # queries += queries_filter
                queries += 2*len(list(set(X) - set(SuT)))*numsamples_m
                S_rounds.append([ele for ele in list(set().union(S, T))])
                time_rounds.append(MPI.Wtime() - p_start)
                query_rounds.append(queries)


                #print ('len(X)', len(X), 'X', X)
                value_randset = estimate_margvalset_parallel(objective, X, SuT, numsamples_m, k, r, \
                                                            randomstate_local, comm, rank, size)
                queries += 2*numsamples_m
                S_rounds.append([ele for ele in list(set().union(S, T))])
                time_rounds.append(MPI.Wtime() - p_start)
                query_rounds.append(queries)

            # Root process draws the random set, computes T, and sends to all other processors.
            if rank == p_root:
                R = list(np.unique( sampleX_wRepl(X, int( np.ceil(np.float(k)/r))) ))
                T = list(set().union(T, R))
            else:
                T = None
            comm.barrier()
            T = comm.bcast(T, root=0)

            SuT = list(set().union(S, T)) 
            f_of_SuT = objective.value( SuT )
            queries += 1
            S_rounds.append([ele for ele in list(set().union(S, T))])
            time_rounds.append(MPI.Wtime() - p_start)
            query_rounds.append(queries)

            #print 'outer_loop_ii', ii, 'inner_loop_iter', inner_loop_iter, 'v_s_X', v_s_X
            inner_loop_iter+=1
            # print( 'outer_loop_ii', ii, 'inner_loop_iter', inner_loop_iter, 'f(S U T)', f_of_SuT)
            # print ('len(S)', len(S), 'len(T)', len(T), 'len(S)+len(T)', len(S)+len(T), \
            #           'len(np.unique(S+T))', len(np.unique(S+T)), 'len(X)', len(X))
            # print ('OUTERWHILE: f_of_SuT', f_of_SuT, '<(eps/20.0)*(OPT - f_of_S))', (eps/20.0)*(OPT - f_of_S))
        S = list(set().union(S, T))

    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start) 
    S_rounds[-1] = [ele for ele in list(set().union(S, T))]
    time_rounds[-1] = time
    query_rounds[-1] = queries

    return objective.value(S), queries, time, S, S_rounds, time_rounds, query_rounds




def amortizedfilteringOPTS(objective, k, eps, numsamples_m):
    '''Wrapper to generate the necessary OPTS and deltas for the theoretical guarantee and run adaptive sampling over them'''
    time0 = datetime.now()
    queries = 0

    best_singleton = np.max([ objective.value([ele]) for ele in objective.groundset ])
    #opts_adsamp = list( np.array( make_I_original(eps, k) ) * best_singleton )
    opts_adsamp = make_I_from_topksingletons(eps, k, val_best_singleton, val_sum_topk)

    f_S = 0
    S_star = []
    for oo, OPT in enumerate(opts_adsamp[::-1]):
        print('amortizedfilteringOPTS main loop ', oo, 'of', len(opts_adsamp))
        fS_given_OPT, queries_OPT, _, S, _, _, _ = amortizedfiltering(objective, k, eps, OPT, numsamples_m)
        if f_S < fS_given_OPT:
            f_S = fS_given_OPT
            S_star = [ele for ele in S]
            print('amortizedfiltering found new best f_S = ', f_S)
        queries += queries_OPT
        if f_S >= (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk:
            print('Stopping early')
            break

    return f_S, queries, (datetime.now()-time0).total_seconds(), S_star




def amortizedfilteringOPTS_parallel(objective, k, eps, numsamples_m, comm, rank, size, allvals, \
                                    stop_if_approx=False, eps_guarantee=0.1, p_root=0, seed=42):
    'Version where we just store final values of F, Time, Queries to make the plots + some iteration counts for loops'
    
    n = len(objective.groundset)

    # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
    #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
    proc_random_state = np.random.RandomState(seed)
    comm.barrier()
    p_start = MPI.Wtime()
    queries = 0
    queries_vec = [n, k, eps] #[]

    S_star = []
    f_S_star = 0
    #I1 = make_I(eps, k)

    #best_ele_val, best_ele_idx = parallel_vals_faster(objective, [ele for ele in objective.groundset], comm, rank, size)
    if allvals is None:
        allvals = parallel_margvals_returnvals(objective, [], [ele for ele in objective.groundset], comm, rank, size)
    queries += len(objective.groundset)
    queries_vec.append(len(objective.groundset))

    val_sum_topk = np.sum( np.sort(allvals)[-k:] )



    I1 = make_I_from_topksingletons(eps, k, np.max(allvals), val_sum_topk)
    len_I1_orig = len(I1)



    finished = False
    iter_outer_loop1 = 0


    if stop_if_approx:
        midI1 = len(I1)-1 # Start at the highest value we want to try for OPT. 
        #This often works, then we are finished in a single iteration of this outer while-loop!
    else:
        midI1 = len(I1) // 2

    while len(I1)>=1 and not finished:
        iter_outer_loop1+=1

        if rank==p_root:
            print ('Commencing Amort.Filt. I1 iter ', iter_outer_loop1, 'of',\
                int( np.ceil( np.log2(len_I1_orig)+2 ) ), ', seconds elapsed =', MPI.Wtime()-p_start)
        finished = len(I1)==1 

        v = I1[midI1]# * best_ele_val

        f_S, queries_I2, time_I2, S_I2, _, _, _ = amortizedfiltering_parallel(objective, \
                                                                            k, \
                                                                            eps, \
                                                                            v, \
                                                                            numsamples_m, \
                                                                            comm, \
                                                                            rank, \
                                                                            size, \
                                                                            p_root, \
                                                                            seed)

        queries +=  queries_I2 
        queries_vec.append(queries_I2)


        if f_S > f_S_star:#objective.value(S_star):
            S_star = [ele for ele in S_I2]
            f_S_star = f_S

        #if f_S < (1.0-1.0/np.exp(1))*v:
        if f_S < (1.0-1.0/np.exp(1))*v:
            # keep lower half I1
            #print( '\n\nLOOKING LEFT IN I1')
            I1 = I1[:midI1]

        else:
            #print ('\n\nLOOKING RIGHT IN I1')
            I1 = I1[midI1:]
            #I1_found = True

        # Break early if we hit our theoretical guarantee. This speeds up the algorithm, 
        # but may possibly result in a lower value f(S) of the solution.

        if stop_if_approx and rank==p_root:
            print('CHECKING APPROXIMATION GUARANTEE:', 'f_S_star', f_S_star, \
                '(1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk', (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk)

        if stop_if_approx and f_S_star >= (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk:
            if rank==p_root:
                print('APPROXIMATION GUARANTEE REACHED. STOPPING EARLY', 'f_S_star', f_S_star, \
                    '(1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk', (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk)
            break

        # Update binary search location
        midI1 = len(I1) // 2



    comm.barrier()
    p_stop = MPI.Wtime()
    time = (p_stop - p_start)

    if rank == p_root:
        print(queries_vec)
        print('len queries_vec -3:', len(queries_vec)-3)
    return f_S_star, queries, time, S_star




def amortizedfiltering(objective, k, eps, OPT, numsamples_m, maxfilteriters=5):
    '''
    Balkanski, Rubinstein, and Singer Amortized Filtering algorithm, SODA 2019
    *** NOTE *** See below for a different implementation used when OPT is not known and several OPTS must be tried to 
    get the theoretical performance guarantee.
    
    INPUTS:
    class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
    int k -- the cardinality constraint (must be k>0)
    float eps -- the error tolerance between 0 and 1
    float OPT -- the value of the optimal solution S*
    int numsamples -- the sample complexity
    int r -- number of rounds
    
    OUTPUTS:
    float f(L) -- the value of the solution
    int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
    float time -- the processing time to optimize the function.
    list S -- the list of elements in the solution
    list of lists S_rounds -- each element is a list containing the solution set Q at the corresponding round
    list time_rounds -- each element is the elapsed number of seconds measured at the completion of the corresponding round in S_rounds.
    list query_rounds -- each element is the elapsed number queries measured at the completion of the corresponding round in L_rounds.
    '''
    check_inputs(objective, k)

    time0 = datetime.now()
    queries = 0
    S = []
    S_rounds = [[]]
    time_rounds = [0]
    query_rounds = [0]
    r = int( np.ceil(20.0/eps) )

    for ii in range(r):
        #print ('SAMPLING WITH OR OUT REPLACEMENT estimatemargval_returnelesifthresh ASK ERIC')
        if rank == p_root and (ii%25 == 0):
            print ('ii', ii, 'of', r, 'len(S)=', len(S), 'k', k, 'f_of_SuT', f_of_SuT)

        X = list(set(objective.groundset) - set(S))
        T = []

        # THIS WHILE LOOP IS SUPPOSED TO TAKE r
        #print(' check this np.ceil(20.0/eps) instead of 1-eps.../r...check the /r too')
        inner_loop_iter = 0
        assert (not bool(set(S) & set(T))) # set S and set T should be disjoint or we will have issues
        assert(len(S)==len(np.unique(S)))
        assert(len(T)==len(np.unique(T)))

        SuT = list(set().union(S, T))
        f_of_SuT = objective.value(SuT)
        f_of_S = objective.value(S)
        queries += 2

        #print (' FOR LOOP ii', ii, 'f_of_SuT', f_of_SuT, '< (eps/20.0)*(OPT - f_of_S) ', #
        # (eps/20.0)*(OPT - f_of_S) , 'len(SuT)', len(SuT), '<k', k)
        while ( ((f_of_SuT-f_of_S) < (eps/20.0)*(OPT - f_of_S) ) and (len(SuT) < k) ):
            #print ('\n\n\n ENTERED OUTER WHILE LOOP')
            #print ('estimating val randset outer while loop')
            value_randset = estimate_margvalset(objective, X, SuT, numsamples_m, k, r)
            queries += 2*numsamples_m

            ele_thresh = (1.0 + eps/2.0)*(1.0 - eps)*(OPT - f_of_SuT)/np.float(k)

            filteriter = 1
            while (filteriter <= maxfilteriters) and(value_randset < ((1.0 - eps) * (OPT - f_of_SuT)/np.float(r))):
                print ('lenX:', len(X), 'f_of_SuT', f_of_SuT, 'value_randset', value_randset, '<', \
                    (1.0 - eps) * (OPT - f_of_SuT)/np.float(r), 'filter_iter', filteriter, 'of', maxfilteriters)
                #print ('estimating X')
                X = estimatemargval_returnelesifthresh(objective, list(set(X)-set(SuT)), SuT, numsamples_m, k, r, ele_thresh)
                queries += 2*len(list(set(X) - set(SuT)))*numsamples_m
                #print ('estimating val randset inner while loop')
                value_randset = estimate_margvalset(objective, X, SuT, numsamples_m, k, r)
                queries += 2*numsamples_m

                S_rounds.append([ele for ele in list(set().union(S, T)) ])
                time_rounds.append((datetime.now() - time0).total_seconds())
                query_rounds.append(queries)

            R = list(np.unique( sampleX_wRepl(X, int( np.ceil(np.float(k)/r))) ))
            T = list(set().union(T, R))

            SuT = list(set().union(S, T)) 
            f_of_SuT = objective.value( SuT )
            queries += 1
            filteriter+=1


            inner_loop_iter+=1
            #print( 'outer_loop_ii', ii, 'inner_loop_iter', inner_loop_iter, 'f(S U T)', f_of_SuT)
            #print ('len(S)', len(S), 'len(T)', len(T), 'len(S)+len(T)', len(S)+len(T), \
            #           'len(np.unique(S+T))', len(np.unique(S+T)), 'len(X)', len(X))
        S = list(set().union(S, T))

    time = (datetime.now() - time0).total_seconds()
    S_rounds[-1] = [ele for ele in S]
    time_rounds[-1] = time
    query_rounds[-1] = queries
    #return S
    return objective.value(S), queries, time, S, S_rounds, time_rounds, query_rounds




# def adapt_sequencing_knowopt_parallel(objective, k, eps, OPT, comm, rank, size, preprocess_add=False, \
#                             lazy_binsearch=False, lazyouterX=False, debug_asserts=False, \
#                             weight_sampling_eps=1.0, sample_threshold=False, lazyskipbinsearch=False, \
#                             allvals=None, OPT_guess_count=1, verbose=True, p_root=0, seed=42):
#     '''
#     Balkanski, Rubinstein, and Singer Adaptive Sequencing algorithm, STOC 2019
#     *** NOTE *** See below for a different implementation used when OPT is not known and several OPTS must be tried to 
#     get the theoretical performance guarantee.
    
#     INPUTS:
#     class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
#     int k -- the cardinality constraint (must be k>0)
#     float eps -- the error tolerance between 0 and 1
#     float OPT -- the value of the optimal solution S*
#     int numsamples -- the sample complexity
#     int r -- number of rounds
#     OUTPUTS:
#     float f(L) -- the value of the solution
#     int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
#     float time -- the processing time to optimize the function.
#     list S -- the list of elements in the solution
#     '''

#     check_inputs(objective, k)
#     p_start = MPI.Wtime()
#     n = len(objective.groundset)

#     randstate = np.random.RandomState(seed)
#     queries = 0
#     time0 = datetime.now()

#     S = []
#     iters = 0
#     t = np.max(allvals)

#     # Initialize lazyouterX
#     if allvals is None:
#         eles_latest_marg_Xdiscards = np.inf*np.ones(n) # Set to inf so we will check all elements a-priori
#     else:
#         eles_latest_marg_Xdiscards = np.copy(allvals)


#     #while len(S) < k and iters < np.ceil(1.0/eps):
#     while len(S) < k and iters < np.ceil( (1.0/eps) * np.log(k/eps) ):
#         #if rank == p_root:
#             #print('iters < np.ceil(1.0/eps) should be amended to larger limit see STOC19 paper')

#         if OPT_guess_count == 1:
#             sample_complexity = compute_sample_complexity_threshold(eps, k, weight_sampling_eps)
#         else:
#             sample_complexity = compute_sample_complexity_threshold_full(eps, k, weight_sampling_eps, n)

#         if sample_threshold and rank==p_root and verbose:
#             print('\n')
#             print('SAMPLE COMPLEXITY:', sample_complexity)
#             print('\n')

#         iters += 1

#         #t_FAST = (1.0-eps)*(OPT - objective.value(S))/np.float(k)
#         t = (1.0-eps)*t
#         # if rank ==0:
#         #     print('t_FAST', t_FAST, 't', t)
#         X_lowval_to_ignore = np.where(eles_latest_marg_Xdiscards < t)[0]
#         X = list(np.sort(list(set(objective.groundset) - set(S)))) 
        
#         if lazyouterX:
#             X = list(set(X) - set(X_lowval_to_ignore))
#             if rank == p_root and verbose:
#                 print('\n')
#                 print('LAZYOUTERX ignored ', len(X_lowval_to_ignore), 'Groundset Elements!', 't=', t)
#             # print([objective.marginalval([ele], S) for ele in X_lowval_to_ignore])
#             # print('eles are:', [ele for ele in X_lowval_to_ignore])


#         while len(X) and len(S)<k:              

#             prev_len_X = len(X)
#             if preprocess_add:

#                 max_x_seq_size = int( np.min(( (k-len(S)), len(X)) ) )             
#                 x_seq = sampleX_randstate(X, max_x_seq_size, randstate)  
#                 S_len = len(S)
#                 eles_to_add_to_S = parallel_pessimistically_add_x_seq(objective, S, x_seq, t, comm, rank, size)
#                 queries += len(x_seq)
                
#                 if debug_asserts:
#                     eles_to_add_to_S0 = [ ele for ii, ele in enumerate(x_seq) \
#                         if (len(S)<k) and (objective.marginalval( [ele],  list(np.sort(list(set().union(S, x_seq[:(ii)])))) ) >= t) ]
#                     assert(np.array_equal(eles_to_add_to_S0, eles_to_add_to_S))

#                 [ S.append(ele) for ele in eles_to_add_to_S ]
#                 queries += len(x_seq)+1
#                 if rank == p_root:
#                     print('ADDED', len(S)-S_len, 'NEW ELEMENTS IN PESSIMISTIC STEP')


#             # Discard low value elements from X again (speeds up the runtime)
#             new_X_setmin_S = list(np.sort(list(set(X)-set(S))))


#             X_margvals = parallel_margvals_returnvals(objective, S, new_X_setmin_S, comm, rank, size)
#             queries += len(new_X_setmin_S) + 1

#             if debug_asserts:
#                 X_margvals0 = [ objective.marginalval([ele], S) for ele in new_X_setmin_S ] 
#                 assert(np.array_equal(X_margvals0, X_margvals))

#             if len(new_X_setmin_S):
#                 #eles_latest_marg_Xdiscards[[np.array(new_X_setmin_S)]] = X_margvals
#                 eles_latest_marg_Xdiscards[np.array(new_X_setmin_S)] = X_margvals

#             X = [ ele for idx, ele in enumerate(new_X_setmin_S) if X_margvals[idx] >= t]
#             #X_margvals_filtered = [ margval for margval in X_margvals if margval >= t]
#             #print('len(X_margvals)', len(X_margvals), 'len(X)', len(X), 'len(X_margvals_filtered)', len(X_margvals_filtered))
#             queries += len(list(set(X)-set(S)))+1


#             # If we added/discarded enough elements with the pessimistic add step, then 
#             # skip the i* sequence search and move to the next round.
#             #if lazyskipbinsearch and len(eles_to_add_to_S) >= eps*prev_len_X:
#             if lazyskipbinsearch and (len(X) <= (1.0-weight_sampling_eps*eps)*prev_len_X):
#                 if rank == p_root:
#                     print('Lazy skipping binary search')
#                 continue

#             else:
#                 max_a_seq_size = min((k-len(S)), len(X))                
#                 a_seq = sampleX_randstate(X, max_a_seq_size, randstate)  

#                 I2 = make_I2_0idx(eps, max_a_seq_size)
#                 # print('\n\n')
#                 # if rank == 0:
#                 #     print('I2 orig', I2, type(I2))
#                 I2 = np.arange(max_a_seq_size+1)
#                 # if rank == 0:
#                 #     print('I2 new', I2, type(I2))
#                 # comm.barrier()

#                 I2_copy = np.copy(I2)

#                 #'MAKING MY OWN I2 this may ruin EVERYTHING'
#                 #I2 = np.arange(0,max_a_seq_size)

#                 i_star_found = False

#                 # Initialize lazy binary search
#                 eles_highest_i_passthresh = -np.inf*np.ones(n)
#                 eles_lowest_i_failthresh = np.inf*np.ones(n)

#                 # Initialize samples to determine whether we pass a particular threshold to check each i:
#                 num_samples = int(np.min((sample_complexity, len(X))))
#                 R = sampleX_randstate(X, num_samples, randstate) 

#                 while not i_star_found:
#                     # Update binary search
#                     midI2 =  len(I2) // 2
#                     i = I2[midI2]
#                     i_star_found = len(I2) == 1 # We do this in the while loop to make binary search check final element in I2
                

#                     # Prepare to skip redundant computations of marginal values
#                     S_u_aseq_i = np.sort(list(set().union(S, a_seq[:i])))
#                     X_setmin_aseq_i = np.sort(list(set(X)-set(S_u_aseq_i)))


#                     # Prepare just a subset of X that intersects with samples R, which we use to estimate |X_i| via sampling.
#                     if sample_threshold:
#                         X_setmin_aseq_i = list(np.sort( list(set(R).intersection(X_setmin_aseq_i)) ))
#                     if rank == p_root and verbose:
#                         print('len(X_setmin_aseq_i)', len(X_setmin_aseq_i), 'sample_threshold=', sample_threshold)


#                     # In binary search, we already know some elements exceed the threshold 
#                     # or cannot exceed the threshold due to submodularity
#                     X_setmin_aseq_i_knowntrue  = [ele for ele in X_setmin_aseq_i if eles_highest_i_passthresh[ele] > i]
#                     X_setmin_aseq_i_knownfalse = [ele for ele in X_setmin_aseq_i if eles_lowest_i_failthresh[ele] < i]

#                     X_setmin_aseq_i_totry = list(np.copy(X_setmin_aseq_i))
#                     if lazy_binsearch:
#                         X_setmin_aseq_i_totry = list(np.sort(list(set(X_setmin_aseq_i)\
#                                                 -set(X_setmin_aseq_i_knowntrue)\
#                                                 -set(X_setmin_aseq_i_knownfalse))))


#                     # Compute X_i, the set of elements with high marginal contribution to the current sequence of length i-1
#                     X_i_totry_margvals = parallel_margvals_returnvals(objective, S_u_aseq_i, X_setmin_aseq_i_totry, comm, rank, size)
#                     X_i = [ele for idx, ele in enumerate(X_setmin_aseq_i_totry) if t <= X_i_totry_margvals[idx] ]
#                     len_X_i_trues = np.float(len(X_i))
#                     queries += len(X_setmin_aseq_i_totry)+1



#                     #print('rank', rank, 'len(X_setmin_aseq_i_totry)', len(X_setmin_aseq_i_totry), \
#                     #'len(X_i_totry_margvals)', len(X_i_totry_margvals), 'len_X_i_trues0', len_X_i_trues0)
#                     # len_X_i_trues = 0
#                     # if len(X_setmin_aseq_i_totry):
#                     #     len_X_i_trues = parallel_X_i_sampleestimate_sum(objective, S_u_aseq_i, X_setmin_aseq_i_totry, \
#                     #t, sample_complexity, randstate, comm, rank, size)


#                     if debug_asserts:
#                         X_i_totry_margvals = [ objective.marginalval([ele], S_u_aseq_i) for ele in X_setmin_aseq_i_totry ]
#                         X_i0 = [ele for idx, ele in enumerate(X_setmin_aseq_i_totry) if t <= X_i_totry_margvals[idx] ]
#                         len_X_i_trues0 = np.float(len(X_i0))
#                         assert(len_X_i_trues==len_X_i_trues0)
#                         assert(np.array_equal(X_i0, X_i))


#                     # If computing |X_i| lazily, add the count of known high value elements
#                     if lazy_binsearch: 
#                         len_X_i_trues = len_X_i_trues + len(X_setmin_aseq_i_knowntrue)


#                     # Checks that lazy binary search matches non-lazy binary search exactly
#                     if debug_asserts and not sample_threshold:
#                         X_i_testing = [True for ele in X_setmin_aseq_i if t <= objective.marginalval([ele], S_u_aseq_i) ]
#                         X_i_testing_eles = [ele for ele in X_setmin_aseq_i if t <= objective.marginalval([ele], S_u_aseq_i) ]
#                         assert(len_X_i_trues==np.sum(X_i_testing))
#                         assert( set(X_i_testing_eles) == set().union(X_i, X_setmin_aseq_i_knowntrue) )


#                     if rank == p_root and verbose:
#                         print('len(X_i)', len(X_i), 'len(X)', len(X), '#knowntrue =',  len(X_setmin_aseq_i_knowntrue), \
#                             '#knownfalse=', len(X_setmin_aseq_i_knownfalse), '#queried=', len(X_i_totry_margvals))

#                     #print('len(I2) is', len(I2), 'len(X_i) is', len(X_i), 'len(X) is', len(X), 'len a_seq is', len(a_seq), \
#                     #'JUST FINISHED checking for i in range',  I2[0], I2[-1] )
#                     if rank == p_root and verbose:
#                         print('i', i, 'len_X_i_trues', len_X_i_trues, 'len(X_setmin_aseq_i)',\
#                             len(X_setmin_aseq_i), '(1.0-eps)*len(X_setmin_aseq_i)', (1.0-eps)*len(X_setmin_aseq_i))

#                     # X_i too small, so need to check the left half
#                     #' THIS IS ME HERE SAYING WE NEED TO (DISCARD + ADD) A CONSTANT FRACTION EACH ITERATION'
#                     #if len(X_i) < (1.0-eps)*len(X):
#                     if (not i_star_found) and (len_X_i_trues < (1.0-2.0*eps)*len(X_setmin_aseq_i)):
#                     #if len(X_i) < (1.0-eps)*len(X_setmin_aseq_i):
#                         I2 = I2[:midI2]
#                         # update lazily
#                         if len(X_i):
#                             #eles_highest_i_passthresh[[np.array(X_i)]] = np.maximum( i*np.ones(len(X_i)), \
#                             #eles_highest_i_passthresh[[np.array(X_i)]] )
#                             eles_highest_i_passthresh[np.array(X_i)] = \
#                                 np.maximum( i*np.ones(len(X_i)), eles_highest_i_passthresh[np.array(X_i)] )
#                         if rank == p_root and verbose:
#                             print('len(I2) is', len(I2), 'len(X_i) is', len(X_i), 'len(X) is', len(X), ' looking left', \
#                                 'len a_seq is', len(a_seq), 'checking for i in range',  I2[0], I2[-1] )

#                     # X_i big enough, so need to check the right half
#                     elif (not i_star_found):
#                         I2 = I2[midI2:]
#                         # update lazily
#                         if rank == p_root and verbose:
#                             print('X_setmin_aseq_i_knownfalse', X_setmin_aseq_i_knownfalse)
#                         fail_eles = list(set(X_setmin_aseq_i_totry) - set(X_i))
#                         if len(fail_eles):
#                             #eles_lowest_i_failthresh[[(fail_eles)]] = np.minimum( i*np.ones(len(fail_eles)), \
#                             #eles_lowest_i_failthresh[[np.array(fail_eles)]] ) 
#                             eles_lowest_i_failthresh[(fail_eles)] = \
#                                 np.minimum( i*np.ones(len(fail_eles)), eles_lowest_i_failthresh[np.array(fail_eles)] ) 

#                         if rank == p_root and verbose:
#                             print('should have updated a maximum of', len(X_setmin_aseq_i_totry) - len(X_i), \
#                                 'i values in eles_lowest_i_failthresh:', \
#                                 'np.sum(eles_lowest_i_failthresh==i)', np.sum(eles_lowest_i_failthresh==i))
#                             print('len(I2) is', len(I2), 'len(X_i) is', len(X_i), 'len(X) is', len(X), ' looking right', \
#                                 'len a_seq is', len(a_seq), 'checking for i in range',  I2[0], I2[-1] )

#                     elif (len_X_i_trues < (1.0-eps)*len(X_setmin_aseq_i)):
#                         # I_star found.
#                         # Final element Did not pass threshold, so final i_star is one left in I2 (or -1 if we are currently at i=0)
#                         i_notpass_idx = np.where(I2[0] == I2_copy)[0][0]
#                         if i_notpass_idx >0:
#                             i_star = I2_copy[i_notpass_idx-1]
#                         else:
#                             i_star = -1

#                         if rank == p_root and verbose:
#                             print('I2[0] did not pass', 'I2[0]=', I2[0], 'I2_copy[np.where(I2[0] == I2_copy)[0][0]]=',\
#                                 I2_copy[np.where(I2[0] == I2_copy)[0][0]], 'np.where(I2[0] == I2_copy)[0][0]=', \
#                                 np.where(I2[0] == I2_copy)[0][0], 'i_star=', i_star)
#                     else:
#                         # I_star found.
#                         # Final element PASSED threshold, so final i_star is remaining element in I2
#                         i_star = I2[0]
                        


#                 # Add the new sequence a_i_star to S

#                 a_seq_to_add = a_seq[:(i_star+1)]
#                 S = list(np.sort(list(set().union(S, a_seq_to_add))))
#                 if rank == p_root and verbose:
#                     print('i_star is', i_star, 'len a_seq is', len(a_seq), 'len(S)=', len(S))
#                 if len(S) == k:
#                     break

#                 X = list(np.sort(list(set(X)-set(S))))

#     #return S_star, I1_found
#     comm.barrier()
#     time = MPI.Wtime() - p_start
#     if rank == p_root:
#         print ('STOC19:', objective.value(S), queries, time, 'with k=', k, 'n=', len(objective.groundset), 'eps=', eps, '|S|=', len(S))
#         print('preprocess_add=', preprocess_add, 'lazy_binsearch=', lazy_binsearch, 'lazyouterX=', lazyouterX, \
#             'debug_asserts=', debug_asserts, 'weight_sampling_eps=', weight_sampling_eps, 'sample_threshold=', sample_threshold)
#     return objective.value(S), queries, time, S#, len_X_i_vec, a_seq_list#, iter_outer_loop, iter_while2, iter_while3, iter_while4




# def adapt_sequencing_guessopt_parallel(objective, k, eps, comm, rank, size, preprocess_add=True, lazy_binsearch=True, \
#                             lazyouterX=True, debug_asserts=False, weight_sampling_eps=1.0, \
#                             sample_threshold=True, lazyskipbinsearch=True, allvals=None, verbose=False, \
#                             stop_if_approx=True, eps_guarantee=0, p_root=0, seed=42):
#     '''
#     Balkanski, Rubinstein, and Singer Adaptive Sequencing algorithm, STOC 2019
#     *** NOTE *** See below for a different implementation used when OPT is not known and several OPTS must be tried to 
#     get the theoretical performance guarantee.
    
#     INPUTS:
#     class objective -- contains the methods 'value()' that we want to optimize and its marginal value function 'marginalval()' 
#     int k -- the cardinality constraint (must be k>0)
#     float eps -- the error tolerance between 0 and 1
#     float OPT -- the value of the optimal solution S*
#     int numsamples -- the sample complexity
#     int r -- number of rounds
#     OUTPUTS:
#     float f(L) -- the value of the solution
#     int queries -- the total queries (marginal values count as 2 queries since f(T)-f(S) )
#     float time -- the processing time to optimize the function.
#     list S -- the list of elements in the solution
#     '''
    
#     n = len(objective.groundset)

#     # Each processor gets its own random state so they can independently draw IDENTICAL random sequences a_seq.
#     #proc_random_states = [np.random.RandomState(seed) for processor in range(size)]
#     proc_random_state = np.random.RandomState(seed)
#     comm.barrier()
#     p_start = MPI.Wtime()
#     queries = 0
#     queries_vec = [n, k, eps] #[]

#     S_star = []
#     f_S_star = 0
#     #I1 = make_I(eps, k)

#     #best_ele_val, best_ele_idx = parallel_vals_faster(objective, [ele for ele in objective.groundset], comm, rank, size)
#     if allvals is None:
#         allvals = parallel_margvals_returnvals(objective, [], [ele for ele in objective.groundset], comm, rank, size)
#     queries += len(objective.groundset)
#     queries_vec.append(len(objective.groundset))

#     val_sum_topk = np.sum( np.sort(allvals)[-k:] )


#     I1 = make_I_from_topksingletons(eps, k, np.max(allvals), val_sum_topk)
#     len_I1_orig = len(I1)


#     finished = False
#     iter_outer_loop1 = 0


#     if stop_if_approx:
#         # Start at the highest value we want to try for OPT. 
#         # This often works, then we are finished in a single iteration of this outer while-loop!
#         midI1 = len(I1)-1 
#     else:
#         midI1 = len(I1) // 2

#     while len(I1)>=1 and not finished:
#         iter_outer_loop1+=1

#         if rank==p_root:
#             print ('Commencing I1 iter ', iter_outer_loop1, 'of', int( np.ceil( np.log2(len_I1_orig)+2 ) ), \
#                     ', seconds elapsed =', MPI.Wtime()-p_start)
#         finished = len(I1)==1 
#         #v = I1[midI1] * best_ele_val
#         v = I1[midI1]# * best_ele_val

#         f_S, queries_I2, time_I2, S_I2 = STOC19_knowopt_parallel( objective, \
#                                                                 k, \
#                                                                 eps, \
#                                                                 v, \
#                                                                 comm,\
#                                                                 rank, \
#                                                                 size, \
#                                                                 preprocess_add, \
#                                                                 lazy_binsearch, \
#                                                                 lazyouterX, \
#                                                                 debug_asserts, \
#                                                                 weight_sampling_eps,\
#                                                                 sample_threshold, \
#                                                                 lazyskipbinsearch, \
#                                                                 allvals, \
#                                                                 OPT_guess_count=iter_outer_loop1, \
#                                                                 verbose=verbose, \
#                                                                 p_root=p_root, \
#                                                                 seed=seed)

#         queries +=  queries_I2 
#         queries_vec.append(queries_I2)


#         if f_S > f_S_star:#objective.value(S_star):
#             S_star = [ele for ele in S_I2]
#             f_S_star = f_S

#         #if f_S < (1.0-1.0/np.exp(1))*v:
#         if f_S < (1.0-1.0/np.exp(1))*v:
#             # keep lower half I1
#             #print( '\n\nLOOKING LEFT IN I1')
#             I1 = I1[:midI1]

#         else:
#             #print ('\n\nLOOKING RIGHT IN I1')
#             I1 = I1[midI1:]
#             #I1_found = True

#         # Break early if we hit our theoretical guarantee. This speeds up the algorithm, 
#         #but may possibly result in a lower value f(S) of the solution.

#         if stop_if_approx and rank==p_root:
#             print('CHECKING APPROXIMATION GUARANTEE:', 'f_S_star', f_S_star, \
#                 '(1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk', (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk)

#         if stop_if_approx and f_S_star >= (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk:
#             if rank==p_root:
#                 print('APPROXIMATION GUARANTEE REACHED. STOPPING EARLY', 'f_S_star', f_S_star, \
#                     '(1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk', (1.0-1.0/np.exp(1)-eps_guarantee)*val_sum_topk)
#             break

#         # Update binary search location
#         midI1 = len(I1) // 2



#     comm.barrier()
#     p_stop = MPI.Wtime()
#     time = (p_stop - p_start)

#     if rank == p_root:
#         print(queries_vec)
#         print('len queries_vec -3:', len(queries_vec)-3)
#     return f_S_star, queries, time, S_star




def parallel_margvals(objective, L, N, comm, rank, size):
    '''
    *NOTE* see faster/alternative implementations below, parallel_margvals_faster()
    Parallel-compute the marginal value f(element)-f(L) of each element in N.
    Returns the tuple of top value, best element
    '''
    comm.barrier()
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    # if rank == p_root:
    #     N_split = np.array_split(N, size)
    # else:
    #     N_split = None

    # Can actually do this locally!
    N_split_local = np.array_split(N, size)[rank]

    # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    ele_vals_local_vals = [ objective.marginalval( [elem], L ) for elem in N_split_local ]
    ele_vals_local_vals_max = np.max(ele_vals_local_vals)
    ele_vals_local_vals_max_ele = N_split_local[ np.argmax(ele_vals_local_vals) ]

    # Reduce the results to all processes
    ele_vals_global_vals_max, ele_vals_global_max_ele = \
        comm.allreduce(sendobj=(ele_vals_local_vals_max, ele_vals_local_vals_max_ele), op=MPI.MAXLOC)

    return ele_vals_global_vals_max, ele_vals_global_max_ele




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




def parallel_margvals_returneles_ifthresh(objective, L, N, threshold, comm, rank, size):
    ''' All processes get the subset of elements in N whose marginal value to set L meets threshold '''
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    comm.barrier()
    N_split_local = np.array_split(N, size)[rank]

    # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    elems_meet_thresh_local = [ elem for elem in N_split_local if objective.marginalval( [elem], L ) >= threshold]
    
    # Gather the partial results to all processes
    elems_meet_thresh_nested = comm.allgather(elems_meet_thresh_local)
    elems_meet_thresh_flattened = [ele for sublist in elems_meet_thresh_nested for ele in sublist] # flattens the list
    return elems_meet_thresh_flattened




def parallel_margvals_forSG(objective, L, N, comm, rank, size):
    '''
    Parallel-compute the marginal value f(element)-f(L) of each element in N. Version for stochasticgreedy_parallel.
    Returns the tuple of top value, best element
    '''
    # NOTE: processors who have empty lists of elements will return -1
    N_split_local = np.array_split(N, size)[rank]

    # Compute the marginal addition for each elem in N, then add the best one to solution L; 
    # then remove it from remaining elements N
    if len(N_split_local):
        ele_vals_local_vals = [ objective.marginalval( [elem], L ) for elem in N_split_local ]
        ele_vals_local_vals_max = np.max(ele_vals_local_vals)
        ele_vals_local_vals_max_ele = N_split_local[ np.argmax(ele_vals_local_vals) ]

    else:
        ele_vals_local_vals_max = -1
        ele_vals_local_vals_max_ele = -1

    ele_vals_global_vals_max, ele_vals_global_max_ele = \
        comm.allreduce(sendobj=(ele_vals_local_vals_max, ele_vals_local_vals_max_ele), op=MPI.MAXLOC)
    assert(ele_vals_local_vals_max_ele != -1)

    return ele_vals_global_vals_max, ele_vals_global_max_ele




def parallel_margvals_faster(objective, L, N, comm, rank, size):
    '''
    Parallel-compute the marginal value f(element)-f(L) of each element in N.
    Returns the tuple of top value, best element
    '''
    comm.barrier()

    N_split_local = np.array_split(N, size)[rank]

    # with local buffering
    #N_split_local = np.empty( np.int( np.ceil(np.float(len(N)) / np.float(size)) ) )
    #N_split_local = [-1] * ( np.int( np.ceil(np.float(len(N)) / np.float(size)) ) )
    #comm.Scatter(sendbuf=N_split, recvbuf=N_split_local, root=p_root)

    # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    if len(N_split_local):
        ele_vals_local_vals = [ objective.marginalval( [elem], L ) for elem in N_split_local ]
        ele_vals_local_vals_max = np.max(ele_vals_local_vals)
        ele_vals_local_vals_max_ele = N_split_local[ np.argmax(ele_vals_local_vals) ]

    else:
        (ele_vals_local_vals_max, ele_vals_local_vals_max_ele) = (-np.inf, -np.inf)
    # Reduce the partial results to the root process
    # no local buffers (slower but simpler)
    ele_vals_global_vals_max, ele_vals_global_max_ele = \
        comm.allreduce(sendobj=(ele_vals_local_vals_max, ele_vals_local_vals_max_ele), op=MPI.MAXLOC)

    #print([ele_vals_global_vals_max, ele_vals_global_max_ele, 'global max val and respective best element'])

    return ele_vals_global_vals_max, ele_vals_global_max_ele




def parallel_vals_faster(objective, N, comm, rank, size):
    ''' Parallel-compute the value f(element) of each element in N '''
    # Scatter the roughly equally sized subsets of remaining elements for which we want values
    comm.barrier()
    N_split_local = np.array_split(N, size)[rank]

    # with local buffering
    #N_split_local = np.empty( np.int( np.ceil(np.float(len(N)) / np.float(size)) ) )
    #N_split_local = [-1] * ( np.int( np.ceil(np.float(len(N)) / np.float(size)) ) )
    #comm.Scatter(sendbuf=N_split, recvbuf=N_split_local, root=p_root)

    # Compute the marginal addition for each elem in N, then add the best one to solution L; 
    # then remove it from remaining elements N
    if len(N_split_local):
        ele_vals_local_vals = [ objective.value( [elem] ) for elem in N_split_local ]
        ele_vals_local_vals_max = np.max(ele_vals_local_vals)
        ele_vals_local_vals_max_ele = N_split_local[ np.argmax(ele_vals_local_vals) ]
    else:
        (ele_vals_local_vals_max, ele_vals_local_vals_max_ele) = (-np.inf, -np.inf)

    # Reduce the partial results to the root process
    ele_vals_global_vals_max, ele_vals_global_max_ele = \
        comm.allreduce(sendobj=(ele_vals_local_vals_max, ele_vals_local_vals_max_ele), op=MPI.MAXLOC)

    #print([ele_vals_global_vals_max, ele_vals_global_max_ele, 'global max val and respective best element'])

    return ele_vals_global_vals_max, ele_vals_global_max_ele




def parallel_val_of_sets(objective, list_of_sets, comm, rank, size):
    ''' Parallel-compute the value f(S) of each set (sublist) in list_of_sets, return ordered list of corresponding values f(S) '''
    # Scatter the roughly equally sized subsets of remaining elements for which we want values
    comm.barrier()

    list_split_local = np.array_split(list_of_sets, size)[rank]
    set_vals_local_vals = [ objective.value( myset ) for myset in list_split_local ]

    # Reduce the partial results to the root process
    set_vals_nested = comm.allgather(set_vals_local_vals)
    set_vals = [item for sublist in set_vals_nested for item in sublist]

    return set_vals




def parallel_X_i_star(objective, S_U_aseq_istar, X, threshold, comm, rank, size):

    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    comm.barrier()

    X_split_local = np.array_split(X, size)[rank]

    X_i_star_local = [ ele for ele in X_split_local if objective.marginalval( [ele], S_U_aseq_istar ) >= threshold ]
    X_i_star_global_nested = comm.allgather(X_i_star_local)
    X_i_star_global = [item for sublist in X_i_star_global_nested for item in sublist]

    return X_i_star_global




def parallel_X_i_count(objective, S_U_aseq_i, X, threshold, comm, rank, size):
    '''Parallel-computes the COUNT of elements in X meet the marg val threshold t'''
    comm.barrier()
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    X_split_local = np.array_split(X, size)[rank]

    X_i_local = np.sum( [ True for ele in X_split_local if objective.marginalval( [ele], S_U_aseq_i ) >= threshold ] )


    X_i_count_global = comm.allreduce(X_i_local, op=MPI.SUM)

    return X_i_count_global




def parallel_X_i_count(objective, S_U_aseq_i, X, threshold, comm, rank, size):
    '''Parallel-computes the COUNT of elements in X meet the marg val threshold t'''
    comm.barrier()
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    X_split_local = np.array_split(X, size)[rank]

    X_i_local_set = [ ele for ele in X_split_local if objective.marginalval( [ele], S_U_aseq_i ) >= threshold ]

    if rank == 0:
        print('this should be empty:', set(S_U_aseq_i).intersection(set(X)), 'threshold', threshold)
        for xx in set(S_U_aseq_i).intersection(set(X)):
            print(  objective.marginalval( [xx], S_U_aseq_i )  )

    X_i_local = len(X_i_local_set)
    
    X_i_count_global = comm.allreduce(X_i_local, op=MPI.SUM)

    return X_i_count_global




def parallel_X_i_sampleestimate(objective, S_U_aseq_i, X, threshold, sample_complexity, proc_random_state, comm, rank, size):
    '''Parallel-computes the COUNT of elements in X meet the marg val threshold t'''
    comm.barrier()

    numsamples = int(np.min((sample_complexity, len(X))))
    # Processors each locally draw the exact same random set
    R = np.sort( proc_random_state.choice(X, numsamples, replace=False) )

    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    R_split_local = np.array_split(R, size)[rank]

    R_sumabovethresh_local = np.sum( [ True for ele in R_split_local if objective.marginalval( [ele], S_U_aseq_i ) >= threshold ] )
    R_mean_above_thresh_global = (1.0/numsamples) * comm.allreduce(R_sumabovethresh_local, op=MPI.SUM)

    return R_mean_above_thresh_global




def parallel_X_i_sampleestimate_sum(objective, S_U_aseq_i, X, threshold, sample_complexity, proc_random_state, comm, rank, size):
    '''Parallel-computes the COUNT of elements in X meet the marg val threshold t'''
    comm.barrier()

    numsamples = int(np.min((sample_complexity, len(X))))
    # Processors each locally draw the exact same random set
    R = np.sort( proc_random_state.choice(X, numsamples, replace=False) )

    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    R_split_local = np.array_split(R, size)[rank]

    R_sumabovethresh_local = np.sum( [ True for ele in R_split_local if objective.marginalval( [ele], S_U_aseq_i ) >= threshold ] )
    R_sum_above_thresh_global = comm.allreduce(R_sumabovethresh_local, op=MPI.SUM)

    return R_sum_above_thresh_global




def compute_sample_complexity_threshold(eps, k, weight_sampling_eps):
    # For sampling to estimate whether |X_i| passes the threshold:
    #firstterm = (3.0 * (1.0/(np.float(weight_sampling_eps)*eps)**2)) / (1.0 - 2.0*eps)
    
    #Fixed FAST delta to 0.025
    # delta=eps
    delta = 0.025


    eps = np.float(weight_sampling_eps)*eps
    firstterm = (2.0+eps) * (1.0/(eps**2)) / (1.0 - 3.0*eps)
    # secterm   = np.log( (1.0/eps) * np.log( (1.0/eps) * np.log(np.float(k) - sizeS) ) )
    #secterm   = np.log( (1.0/eps) * np.log( (1.0/eps) * np.max(( np.log(2), np.log(np.float(k) - sizeS) )) ) )
    
    #delta not used for computation / onlyt eps used
    # secterm   = np.log( 2.0/eps ) 
    secterm   = np.log( 2.0/delta ) 
    return int(np.ceil( firstterm * secterm ))




def compute_sample_complexity_threshold_full(eps, k, weight_sampling_eps, n):
    # For sampling to estimate whether |X_i| passes the threshold:
    #firstterm = (3.0 * (1.0/(np.float(weight_sampling_eps)*eps)**2)) / (1.0 - 2.0*eps)
    
    # delta = eps
    delta = 0.025


    eps5 = np.float(weight_sampling_eps)*eps
    firstterm = (2.0+eps5) * (1.0/(eps5**2)) / (1.0 - 3.0*eps5)
    # secterm   = np.log( (1.0/eps) * np.log( (1.0/eps) * np.log(np.float(k) - sizeS) ) )
    #secterm   = np.log( (1.0/eps) * np.log( (1.0/eps) * np.max(( np.log(2), np.log(np.float(k) - sizeS) )) ) )

    r1 = np.log(np.log(np.float(k))/eps)
    r2 = 1.0/eps
    r3 = np.log(np.float(n))/eps5

    secterm   = np.log( 2.0*r1*r2*r3/delta ) 

    return int(np.ceil( firstterm * secterm ))




def parallel_pessimistically_add_x_seq(objective, S, x_seq, t, comm, rank, size):
    ''' All processes get the subset of elements i in x_seq whose \
    marginal value to set { S U x_seq_0...x_seq_{i-1} } meets threshold '''
    # Scatter the roughly equally sized subsets of remaining elements for which we want marg values
    comm.barrier()
    #x_seq_split_local = np.array_split(x_seq, size)[rank]
    ii_split_local = np.array_split(list(range(len(x_seq))), size)[rank]

    # Compute the marginal addition for each elem in N, then add the best one to solution L; remove it from remaining elements N
    # [ x_seq[ii] for ii in ii_split_local]
    # ii = ii_split_local[0]
    # list(np.sort(list(set().union(S, x_seq[:(ii)])))) 
    # objective.marginalval( [x_seq[ii]], list(np.sort(list(set().union(S, x_seq[:(ii)])))) ) >= t
    elems_meet_thresh_local = [ x_seq[ii] for ii in ii_split_local \
                                if objective.marginalval( [x_seq[ii]], list(np.sort(list(set().union(S, x_seq[:(ii)])))) ) >= t]
    
    # Gather the partial results to all processes
    elems_meet_thresh_nested = comm.allgather(elems_meet_thresh_local)
    elems_meet_thresh_flattened = [ele for sublist in elems_meet_thresh_nested for ele in sublist] # flattens the list
    return elems_meet_thresh_flattened




def run_alg_all_k_save_data(k_vals_vec, algorithm, my_kwargs, filepath_string, experiment_string, algostring):
    '''Run algorithm (using its my_kwargs) over each value of k in k_vals_vec'''
    f_vec = []
    queries_vec = []
    time_vec = []

    for k_i in k_vals_vec:
        f_k, queries_k, time_k = algorithm(k=k_i, **my_kwargs)
        f_vec.append(f_k)
        queries_vec.append(queries_k)
        time_vec.append(time_k)

    # if want to progressive save inside the loop, make sure to do k_vals_vec[:ii+1] and 
    # change the for k_i... to for ii in range(len(k_vals_vec))
    np.savetxt(filepath_string + experiment_string +'_'+ algostring +".csv", \
                np.vstack((f_vec, queries_vec, time_vec, k_vals_vec)).T, delimiter=",")
    print (experiment_string)
    print (algorithm, algostring)
    print ('f_vec', f_vec)
    print ('queries_vec', queries_vec)
    print ('time_vec', time_vec)




def run_alg_all_k_progsave_data(k_vals_vec, algorithm, my_kwargs, experiment_string, algostring):
    '''Run algorithm (using its my_kwargs) over each value of k in k_vals_vec. 
    This version saves progressively after each run'''
    f_vec = []
    queries_vec = []
    time_vec = []

    for ii, k_i in enumerate(k_vals_vec):
        f_k, queries_k, time_k = algorithm(k=k_i, **my_kwargs)
        f_vec.append(f_k)
        queries_vec.append(queries_k)
        time_vec.append(time_k)

        # if want to progressive save inside the loop, make sure to do k_vals_vec[:ii+1] and 
        # change the for k_i... to for ii in range(len(k_vals_vec))
        np.savetxt(filepath_string + experiment_string +'_'+ algostring +".csv", \
            np.vstack((f_vec, queries_vec, time_vec, k_vals_vec[:(ii+1)])).T, delimiter=",")
    print (experiment_string)
    print (algorithm, algostring)
    print( 'f_vec', f_vec)
    print( 'queries_vec', queries_vec)
    print( 'time_vec', time_vec)







