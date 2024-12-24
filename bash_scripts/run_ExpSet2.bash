pyfile=$1
#echo "mpirun -np ${nthreads} python3 -W ignore ${pyfile}"
# mpirun -np ${nthreads} python3 -W ignore $1 $3 $4


#ls -al
alg5="FAST1"
alg4="PGB2"
prefix=" experiment_results_output_data/"
under="_"
suffix=".csv"

declare -a nT=(1 2 4 8 16 32 64)

declare -a data_name=("BA_100k" "WS_100k" "IMAGESUMM" "TWITTERSUMM")
#declare -a data_name=("WS_100k" "BA_100k" "TWITTERSUMM")
#declare -a data_name=("IMAGESUMM" "TWITTERSUMM")
declare -a objs=("BA" "WS" "IS_PGB" "TS_PGB")
#declare -a objs=("IS_FAST" "TS_FAST")

for i in ${!data_name[@]};
do
	data=${data_name[$i]}
	obj=${objs[$i]}
    for j in ${!nT[@]};
    do
        nthreads=${nT[$j]}
        cmd="mpirun --oversubscribe -np ${nthreads} python3 -W ignore ${pyfile} ${obj} ALL"
        echo $cmd
      	$cmd
    done
    echo "Data generated for '${data}'"
    cat ${prefix}${data}_exp2_PGB_* > ${prefix}${data}_exp2_PGB_tmp.csv
    cat ${prefix}${data}_exp2_FAST_* > ${prefix}${data}_exp2_FAST_tmp.csv
    sed '1!{/^f_of/d;}' ${prefix}${data}_exp2_PGB_tmp.csv > ${prefix}${data}_exp2_PGB.csv
    sed '1!{/^f_of/d;}' ${prefix}${data}_exp2_FAST_tmp.csv > ${prefix}${data}_exp2_FAST.csv
    rm ${prefix}${data}_exp2_PGB_*
    rm ${prefix}${data}_exp2_FAST_*
done

declare -a data_name=("IMAGESUMM" "TWITTERSUMM")
declare -a objs=("IS_FAST" "TS_FAST")

for i in ${!data_name[@]};
do
        data=${data_name[$i]}
        obj=${objs[$i]}
    for j in ${!nT[@]};
    do
        nthreads=${nT[$j]}
        cmd="mpirun -np ${nthreads} python3 -W ignore ${pyfile} ${obj} ALL"
        echo $cmd
        $cmd
    done
    echo "Data generated for '${data}'"
    cat ${prefix}${data}_exp2_PGBVanilla_* > ${prefix}${data}_exp2_PGBVanilla_tmp.csv
    cat ${prefix}${data}_exp2_FAST_* > ${prefix}${data}_exp2_FAST_tmp.csv
    sed '1!{/^f_of/d;}' ${prefix}${data}_exp2_PGBVanilla_tmp.csv > ${prefix}${data}_exp2_PGBVanilla.csv
    sed '1!{/^f_of/d;}' ${prefix}${data}_exp2_FAST_tmp.csv > ${prefix}${data}_exp2_FAST.csv
    rm ${prefix}${data}_exp2_PGB_*
    rm ${prefix}${data}_exp2_FAST_*
done

bash ./plot_scripts/plot_exp2.bash
