#WS
data='WS_100k'
python3 plot_scripts/plot_ExpSet2_T.py v 1 experiment_results_output_data/${data}_exp2_FAST.csv experiment_results_output_data/${data}_exp2_PGB.csv
python3 plot_scripts/plot_ExpSet2_T.py t 1 experiment_results_output_data/${data}_exp2_FAST.csv experiment_results_output_data/${data}_exp2_PGB.csv

#BA
data='BA_100k'
python3 plot_scripts/plot_ExpSet2_T.py v 1 experiment_results_output_data/${data}_exp2_FAST.csv experiment_results_output_data/${data}_exp2_PGB.csv
python3 plot_scripts/plot_ExpSet2_T.py t 1 experiment_results_output_data/${data}_exp2_FAST.csv experiment_results_output_data/${data}_exp2_PGB.csv

#IMAGESUMM
data='IMAGESUMM'
python3 plot_scripts/plot_ExpSet2_T.py v 1 experiment_results_output_data/${data}_exp2_FAST.csv experiment_results_output_data/${data}_exp2_PGB.csv
python3 plot_scripts/plot_ExpSet2_T.py t 1 experiment_results_output_data/${data}_exp2_FAST.csv experiment_results_output_data/${data}_exp2_PGB.csv

#TWITTERSUMM
data='TWITTERSUMM'
python3 plot_scripts/plot_ExpSet2_T.py v 1 experiment_results_output_data/${data}_exp2_FAST.csv experiment_results_output_data/${data}_exp2_PGB.csv
python3 plot_scripts/plot_ExpSet2_T.py t 1 experiment_results_output_data/${data}_exp2_FAST.csv experiment_results_output_data/${data}_exp2_PGB.csv

mogrify -format pdf -- plots/*exp2.png
rm plots/*exp2.png
