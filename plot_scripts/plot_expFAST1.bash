
#ER
data='ER'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv

#WS
data='WS'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv

#BA
data='BA'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv


#SBM
data='SBM'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv

#TRF
data='CAROAD'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv

#IFM
data='INFMAXCalTech'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv

#YOUTUBE2000
data='YOUTUBE50'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv

#MOVIECOVERsubset
data='MOVIECOVERsubset'
python3 plot_scripts/plot_ExpSet1.py v 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py t 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv
python3 plot_scripts/plot_ExpSet1.py q 1 experiment_results_output_data/${data}_PGB.csv experiment_results_output_data/${data}_EXMAX.csv experiment_results_output_data/${data}_BINSEARCHMAX.csv


mogrify -format pdf -- plots/*.png
# rm plots/*exp1.png




