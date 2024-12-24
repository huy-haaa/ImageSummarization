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
datanames=[]
nalgs=0;
for i in range(3, len(sys.argv)):
    arg=sys.argv[i];
    fnames.append( arg );
    pos = arg.rfind('_');
    arg2 = arg[pos+1:]
    pos = arg2.find('.csv');
    alg = arg2[0:pos];
    if(alg=="PGB"):
        alg="LS+PGB"
    if(alg=="PGBVanilla"):
        alg="TOP1+PGB"   
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
    

normalizeX=False;

plot_dir = "plots/"
if (sys.argv[2][0] == '1'):
    printlegend=True;
else:
    printlegend=False;
    
scaleDiv=1;
normalize=False;

postfix='_exp1.png';

colors = [ 'k',
           'y',
           'b',
           'g',
           'r',
           'm',
           'c',
           'gold',
           'deepskyblue',
           'lime',
           'mediumpurple',
           'orange'
            ];

markers = [ 'p',
            's',
            'v',
            '<',
            '>',
            '^',
            '*',
            '1',
            'd',
            'X',
            'o',
            'P' ];
if (sys.argv[1][0] == 'v'):
    outFName= plot_dir + 'val-' +  dataname + postfix;
    colObj = 0;
    normalize=True;
    which=0;
else:
    if (sys.argv[1][0] == 'q'):
        which = 1; #1 = queries
        outFName= plot_dir + 'query-' + dataname + postfix;
        colObj = 1;
    else:
        if (sys.argv[1][0] == 't'):
            which = 3; #1 = time
            outFName= plot_dir +'time-'  + dataname + postfix;
            colObj = 2



X = [];
Obj = [];
ObjStd = [];
skip = [ False for i in range(0,nalgs) ];
nodes = 0;
kmin=0;
for i in range( 0, nalgs ):
    Obj_tmp = []
    ObjStd_tmp = []
    X_tmp = []
    # fname = "/Users/tonmoydey/Documents/Research_TheoreticalAlgo/Code/python-submodular/experiment_results_output_data/ER_100k_FLS.csv"
    fname = fnames[ i ];
    if (os.path.isfile( fname )):
        skip[i]=False;
        print ("Reading from file", fname);
        data = pd.read_csv(fname)  
        k_distinct = data.k.unique()
        k = list(data.k)
        f_of_S = list(data.f_of_S)
        qry = list(data.Queries)
        time = list(data.Time)
        nodes = list(data.n)[0]
        for j in range(len(k_distinct)):
            X_tmp.append(k_distinct[j])
            if(which==0):
                obj_ele = [elem for ii, elem in enumerate(f_of_S) if k_distinct[j] == k[ii]]
            if(which==1):
                obj_ele = [elem for ii, elem in enumerate(qry) if k_distinct[j] == k[ii]]
            if(which==3):
                obj_ele = [elem for ii, elem in enumerate(time) if k_distinct[j] == k[ii]]
            obj_mean = np.mean(obj_ele)
            obj_std = np.std(obj_ele)
            if(i!=0 and which == 0):
                obj_mean = obj_mean/Obj[0][j]
                Obj_tmp.append(obj_mean)
                obj_std = obj_std/Obj[0][j]
                ObjStd_tmp.append(obj_std)
            else:
                Obj_tmp.append(obj_mean)
                ObjStd_tmp.append(obj_std)
                
    Obj.append(Obj_tmp)
    ObjStd.append(ObjStd_tmp)
    X.append(X_tmp)
    
if(len(Obj[0])==0):
    exit()
print(Obj)
  
        

   
plt.gcf().clear();
plt.rcParams['pdf.fonttype'] = 42;
plt.rcParams.update({'font.size': 25});
plt.rcParams.update({'font.weight': "bold"});
plt.rcParams["axes.labelweight"] = "bold";
# plt.xscale('log');
title = str(dataname2) + " (n=" + str(nodes) + ")"
plt.title(title)


#plt.ticklabel_format(axis='both', style='sci' );
if (which == 1):
    print (nodes);
    plt.ylabel( 'Queries / $n$' );
    plt.yscale('log');
    for i in range( 0, nalgs ):
            for j in range( 0, len( Obj[ i ] ) ):
                Obj[i][j] = Obj[i][j]/nodes;
                ObjStd[i][j] = ObjStd[i][j]/nodes;
else:
    if (which == 3):
            # plt.ylabel( 'Adaptive Rounds / $n$' );
            plt.ylabel( 'Time Taken (s)' );
            plt.yscale('log');
            for i in range( 1, nalgs ):
                for j in range( 0, len( Obj[ i ] ) ):
                    Obj[i][j] = Obj[i][j];
                    ObjStd[i][j] = ObjStd[i][j];
                    # Obj[i][j] = Obj[i][j] / X[i][j];
                    # ObjStd[i][j] = ObjStd[i][j] / X[i][j];
    else:
        plt.ylabel( "Objective / Greedy" )



plt.xlabel( "$k / n$" );
for i in range( 1, nalgs ):
    for j in range( 0, len( X[ i ] ) ):
        X[i][j] = X[i][j] / nodes;
# if normalizeX:
#     plt.xlabel( "$k / n$" );
#     for i in range( 1, nalgs ):
#         for j in range( 0, len( X[ i ] ) ):
#             X[i][j] = X[i][j] / nodes;
# else:
#     plt.xlabel( '$k$' );

plt.xscale('log')
    
if which==0 and normalize==False:
    #normalize by nodes
    for i in range(0,nalgs):
        for j in range( 0, len( Obj[ i ] ) ):
            Obj[i][j] = Obj[i][j];
            ObjStd[i][j] = ObjStd[i][j];
            plt.ylabel( "Objective / Greedy" );            

markSize=20;

if normalize:
    algmin = 1;
    algmax = nalgs;
    plt.ylim( 0.50, 1.05 );
else:
    algmin = 1;
    algmax = nalgs;

plt.xlim( 0.0005,0.15  );
print( nodes );

if (which == 1):
    #plt.axhline( nodes, color='r' );
    ax = plt.gca();
    #ax.annotate('$n$', xy=(float(kmin), nodes), xytext=(float(kmin)-75, nodes - 100000), size=15 );

for i in range(algmin,algmax):
    mi = 1;
    plt.plot( X[i], Obj[i], ':', marker=markers[i],  label=algnames[i],ms = markSize,color = colors[i], markevery = mi);
    BObj = np.asarray( Obj[i] );
    BObjStd = np.asarray( ObjStd[i] );
    if i != 3:
        plt.fill_between( X[i], BObj - BObjStd, BObj + BObjStd,
                          alpha=0.5, edgecolor=colors[i], facecolor=colors[i]);


#plt.errorbar( X, Obj, yerr=BObjStd, fmt='-');



plt.gca().grid(which='major', axis='both', linestyle='--')


#plt.grid(color='grey', linestyle='--' );
if printlegend:
    plt.legend(loc='best', numpoints=1,prop={'size':18},framealpha=0.6);

plt.savefig( outFName, bbox_inches='tight', dpi=400 );
#plt.legend(loc='best', numpoints=1,prop={'size':18},framealpha=1.0);
#plt.savefig( 'WithLegend.png', bbox_inches='tight',dpi=500 );
