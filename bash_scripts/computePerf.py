import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import statistics
import os.path
import pandas as pd
import statistics

which=0; # 0 = solValue
if (len(sys.argv) == 1):
    print("usage: plot [vqm] [printLegend:1 or 0] <results filenames>");
    exit();

###parse arguments
fnames=[];
algnames=[];
datanames = []
nalgs=0;

for i in range(2, len(sys.argv)):
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
    if(dataname == "BA"):
        dataname2 = "MaxCover(BA)"
    if(dataname == "ER"):
        dataname2 = "MaxCover(ER)"
    if(dataname == "WS"):
        dataname2 = "MaxCover(WS)"
    if(dataname == "TWITTERSUMM"):
        dataname2 = "TwitterSumm"
    if(dataname == "IMAGESUMM"):
        dataname2 = "ImageSumm"
    if(dataname == "INFLUENCEEPINIONS"):
        dataname2 = "InfluenceMax"
    if(dataname == "YOUTUBE2000"):
        dataname2 = "RevenueMax"
    if(dataname == "CAROADFULL"):
        dataname2 = "TrafficMonitor"
    datanames.append(dataname2)
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
    # print(i,fname)
    if (os.path.isfile( fname )):
        skip[i]=False;
        # print ("Reading from file", fname);
        data = pd.read_csv(fname)  
        k_distinct = data.k.unique()
        if(i%2!=0):
            total_k += len(k_distinct)
            # print("\nApp\t", "\tk\t","\tFAST\t" ,"\t\tPGB\t", "\t\tSpeedUp(%)")
        k = list(data.k)
        f_of_S = list(data.f_of_S)
        qry = list(data.Queries)
        time = list(data.Time)
        nodes = list(data.n)[0]
        
        for j in range(len(k_distinct)):
            X_tmp.append(k_distinct[j])
            if(which==0):
                if (sys.argv[1][0] == 'v'):
                    obj_ele = [elem for ii, elem in enumerate(f_of_S) if k_distinct[j] == k[ii]]
                if (sys.argv[1][0] == 't'):
                    obj_ele = [elem for ii, elem in enumerate(time) if k_distinct[j] == k[ii]]
                if (sys.argv[1][0] == 'q'):
                    obj_ele = [elem for ii, elem in enumerate(qry) if k_distinct[j] == k[ii]]
                
            obj_mean = np.mean(obj_ele)
            if(i%2!=0 and which == 0):
                # print("App: ", datanames[i], "k: ",k_distinct[j]," " , Obj[i-1][j], obj_mean, (Obj[i-1][j] - obj_mean)*100/Obj[i-1][j])
                # print(datanames[i], "\t",k_distinct[j],"\t" , Obj[i-1][j], "\t",obj_mean,"\t", (Obj[i-1][j] - obj_mean)*100/Obj[i-1][j])
                
                # obj_mean = (obj_mean / Obj[i-1][j])
                
                if (sys.argv[1][0] == 'v'):
                    if(obj_mean >= Obj[i-1][j]):
                        count_btr += 1
                    # else:
                    #    print("FAST better: ", datanames[i], " k: ", k_distinct[j], " PGB: ", obj_mean, " FAST: ",Obj[i-1][j]) 
                else:
                    if(obj_mean <= Obj[i-1][j]):
                        count_btr += 1
                    # else:
                    #    print("FAST better: ", datanames[i], " k: ", k_distinct[j], " PGB: ", obj_mean, " FAST: ",Obj[i-1][j])

                Obj_tmp.append(obj_mean)
            else:
                Obj_tmp.append(obj_mean)
                
    Obj.append(Obj_tmp)
    ObjStd.append(Obj_tmp)
    
if(len(Obj[0])==0):
    exit()
print("\n")
final_Obj_1=[]
final_Obj_2=[]
# printing Aligned Header
if (sys.argv[1][0] == 'v'):
    print(f"{'data' : <25}{'Avg PGB/FAST' : <25}{'Mean PGB' : <25}{'Mean FAST' : <5}")#{'Min PGB/FAST' : ^25}{'Max PGB/FAST' : >5}")
else:
    print(f"{'data' : <25}{'Avg FAST/PGB' : <25}{'Mean PGB' : <25}{'Mean FAST' : <5}")#{'Min FAST/PGB' : ^25}{'Max FAST/PGB' : >5}")
  
# printing values of variables in Aligned manner
for i in range(len(datanames)):
    if(i%2!=0):
        final_Obj_1 = np.append(final_Obj_1,Obj[i-1])
        final_Obj_2 = np.append(final_Obj_2,Obj[i])
        if (sys.argv[1][0] == 'v'):
            print(f"{datanames[i] : <25}{np.sum(Obj[i])/np.sum(Obj[i-1]) : <25}{np.mean(Obj[i]) : <25}{np.mean(Obj[i-1]) : <5}") #{np.min(Obj[i]) : ^25}{np.max(Obj[i]) : >5}")
        else:
            print(f"{datanames[i] : <25}{np.sum(Obj[i-1])/np.sum(Obj[i]) : <25}{np.mean(Obj[i]) : <25}{np.mean(Obj[i-1]) : <5}") 
        # print("data: ", datanames[i], "\tAvg PGB/FAST: ", np.mean(Obj[i]), "\tMin PGB/FAST: ", np.min(Obj[i]), "\tMax PGB/FAST: ", np.max(Obj[i]))
# print(final_Obj.shape)



if (sys.argv[1][0] == 'v'):
    final_obj = final_Obj_2/final_Obj_1
    print("\nOverall Avg PGB/FAST: ",np.sum(final_Obj_2)/np.sum(final_Obj_1) )
    # print("Overall Min PGB/FAST: ",np.min(final_Obj) )
    # print("Overall Max PGB/FAST: ",np.max(final_Obj) )
else:
    final_obj = final_Obj_1/final_Obj_2
    print("\nOverall Avg FAST/PGB: ",np.sum(final_Obj_1)/np.sum(final_Obj_2) )
    # print("Overall Min FAST/PGB: ",np.min(final_Obj) )
    # print("Overall Max FAST/PGB: ",np.max(final_Obj) )
# print(final_obj)
# print("\nOverall StdDev Improvement: ",np.std(final_Obj) )
print("\nHarmonic Mean is % s " % (statistics.harmonic_mean(final_obj)))
print("\nNo. of Better Scenarios: ", count_btr, " Total Scenarios: ", total_k, "  Percantage: ", count_btr*100/total_k, "%")
    