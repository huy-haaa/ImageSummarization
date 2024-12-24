import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import statistics
import os.path
import pandas as pd

which=0; # 0 = solValue
if (len(sys.argv) == 1):
    print("usage: plot [vqm] [printLegend:1 or 0] <results filenames>");
    exit();

###parse arguments
fnames=[];
algnames=[];
datanames = []
nalgs=0;

for i in range(1, len(sys.argv)):
    arg=sys.argv[i];
    fnames.append( arg );
    pos = arg.rfind('_');
    arg2 = arg[pos+1:]
    pos = arg2.find('.csv');
    alg = arg2[0:pos];
    if(alg=="PGB"):
        alg="LS+PGB"
    # print("alg:", alg)
    algnames.append( alg );
    nalgs = nalgs + 1;
    pos = arg.find("/");
    arg = arg[pos+1:];
    pos = arg.find('_');
    dataname=arg[0:pos];
    datanames.append(dataname)
        
# print(datanames)
# print(algnames)


X = [];
Obj = [];
ObjStd = [];
skip = [ False for i in range(0,nalgs) ];
nodes = 0;
kmin=0;
which=0
count_btr = 0
total_k = 0

if (sys.argv[1][0] == 'v'):
    print ("\n\nRESULTS FOR OBJECTIVE");
if (sys.argv[1][0] == 't'):
    print ("\n\nRESULTS FOR PARALLEL RUNTIME");
if (sys.argv[1][0] == 'q'):
    print ("\n\n\nRESULTS FOR NUMBER OF QUERY CALLS");

for i in range( 0, nalgs, 1 ):
    Obj_tmp = []
    ObjStd_tmp = []
    X_tmp = []
    # fname = "/Users/tonmoydey/Documents/Research_TheoreticalAlgo/Code/python-submodular/experiment_results_output_data/ER_100k_FLS.csv"
    fname = fnames[ i ];
    print(i,fname)
    if (os.path.isfile( fname )):
        skip[i]=False;
        # print ("Reading from file", fname);
        data = pd.read_csv(fname)  
        k_distinct = data.k.unique()
        nproc_distinct = data.nproc.unique()
        nproc_distinct = np.sort(nproc_distinct)
        nproc = list(data.nproc)
        if(i%2!=0):
            total_k += len(nproc_distinct)
            # print("\nApp\t", "\tk\t","\tFAST\t" ,"\t\tPGB\t", "\t\tSpeedUp(%)")
        k = list(data.k)
        time = list(data.Time)
        nodes = list(data.n)[0]
        # print(nproc_distinct)
        for j in range(len(nproc_distinct)):
            X_tmp.append(nproc_distinct[j])
            if(which==0):
                obj_ele = [elem for ii, elem in enumerate(time) if nproc_distinct[j] == nproc[ii]]
            # print(i)    
            obj_mean = np.mean(obj_ele)
            if(i%2!=0 and which == 0):
                # print("App: ", datanames[i], "k: ",k_distinct[j]," " , Obj[i-1][j], obj_mean, (Obj[i-1][j] - obj_mean)*100/Obj[i-1][j])
                # print(datanames[i], "\t",k_distinct[j],"\t" , Obj[i-1][j], "\t",obj_mean,"\t", (Obj[i-1][j] - obj_mean)*100/Obj[i-1][j])
                
                # obj_mean = (obj_mean / Obj[i-1][j])
                obj_mean = (Obj[i-1][j] / obj_mean)

                if(obj_mean > 1):
                    count_btr += 1
                                
                Obj_tmp.append(obj_mean)
            else:
                Obj_tmp.append(obj_mean)
                # print(obj_mean)
    Obj.append(Obj_tmp)
    ObjStd.append(Obj_tmp)
    
if(len(Obj[0])==0):
    exit()
print("\n")
final_Obj=[]
# printing Aligned Header
if (sys.argv[1][0] == 'v'):
    print(f"{'data' : <25}{'Avg PGB/FAST' : ^25}{'Min PGB/FAST' : ^25}{'Max PGB/FAST' : >5}")
else:
    print(f"{'data' : <25}{'Avg FAST/PGB' : ^25}{'Min FAST/PGB' : ^25}{'Max FAST/PGB' : >5}")
  
# printing values of variables in Aligned manner
for i in range(len(datanames)):
    if(i%2!=0):
        final_Obj = np.append(final_Obj,Obj[i])
        print(f"{datanames[i] : <25}{np.mean(Obj[i]) : ^25}{np.min(Obj[i]) : ^25}{np.max(Obj[i]) : >5}")
        # print("data: ", datanames[i], "\tAvg PGB/FAST: ", np.mean(Obj[i]), "\tMin PGB/FAST: ", np.min(Obj[i]), "\tMax PGB/FAST: ", np.max(Obj[i]))
# print(final_Obj.shape)

if (sys.argv[1][0] == 'v'):
    print("\nOverall Avg PGB/FAST: ",np.mean(final_Obj) )
    print("Overall Min PGB/FAST: ",np.min(final_Obj) )
    print("Overall Max PGB/FAST: ",np.max(final_Obj) )
else:
    print("\nOverall Avg FAST/PGB: ",np.mean(final_Obj) )
    print("Overall Min FAST/PGB: ",np.min(final_Obj) )
    print("Overall Max FAST/PGB: ",np.max(final_Obj) )

# print("\nOverall StdDev Improvement: ",np.std(final_Obj) )
print("\nNo. of Better Scenarios: ", count_btr, " Total Scenarios: ", total_k, "  Percantage: ", count_btr*100/total_k, "%")
    