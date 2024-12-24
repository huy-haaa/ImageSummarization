nthreads=$2
pyfile=$1
#echo "mpirun -np ${nthreads} python3 -W ignore ${pyfile}"
# mpirun -np ${nthreads} python3 -W ignore $1 $3 $4


prefix="experiment_results_output_data/"
under="_"
suffix=".csv"

declare -a data_name=("CAROADFULL" "INFLUENCEEPINIONS" "TWITTERSUMM" "YOUTUBE2000" "ER_100k" "WS_100k" "BA_100k"  "IMAGESUMM")

declare -a objs=("TRF" "IFM" "TS" "RVM" "ER" "WS" "BA" "IS_PGB")

for i in ${!data_name[@]};
do
	data=${data_name[$i]}
	obj=${objs[$i]}
	cmd="mpirun -np ${nthreads} python3 -W ignore ${pyfile} ${obj} ALL"
	echo $cmd
	$cmd
    
    echo "Data generated for '${data}'"
done

declare -a data_name=("IMAGESUMM" "IMAGESUMM" "IMAGESUMM")

declare -a objs=("IS_FAST") #"IS_LS" "IS_PLG" 

for i in ${!data_name[@]};
do
	data=${data_name[$i]}
	obj=${objs[$i]}
	cmd="mpirun -np ${nthreads} python3 -W ignore ${pyfile} ${obj} ALL"
	echo $cmd
	$cmd
    
    echo "Data generated for '${data}'"
done

bash ./plot_scripts/plot_exp1.bash
