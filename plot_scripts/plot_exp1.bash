
#ER
data='ER_100k'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv

#WS
data='WS_100k'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv

#BA
data='BA_100k'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
#TRF
data='CAROADFULL'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv

#IFM
data='INFLUENCEEPINIONS'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv

#YOUTUBE2000
data='YOUTUBE2000'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv

#IMAGESUMM
data='IMAGESUMM'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv

#TWITTERSUMM
data='TWITTERSUMM'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_exp1_PLG.csv experiment_results_output_data/${data}_exp1_FAST.csv experiment_results_output_data/${data}_exp1_PGB.csv


mogrify -format pdf -- plots/*exp1.png
rm plots/*exp1.png

##SBM
#data='SBM'
#python3 plot_ExpSet1.py v 1 experiment_results_output_data/${data}_100k_PLG.csv experiment_results_output_data/${data}_100k_FAST1.csv experiment_results_output_data/${data}_100k_PGB2.csv
#python3 plot_ExpSet1.py t 1 experiment_results_output_data/${data}_100k_PLG.csv experiment_results_output_data/${data}_100k_FAST1.csv experiment_results_output_data/${data}_100k_PGB2.csv
