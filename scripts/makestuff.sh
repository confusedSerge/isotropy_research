sde=$1
wordsim_de=$2
freq_de=$3

puk=$4
wordsim_eng=$5
freq_eng=$6

# GER
python3.7 src/floating_cent_newmes_l2n.py $sde $wordsim_de $freq_de >> sde_d300_nm_l2n.tsv
python3.7 src/floating_cent_newmes_nol2n.py $sde $wordsim_de $freq_de >> sde_d300_nm_nol2n.tsv

# ENG
python3.7 src/floating_cent_newmes_l2n.py $puk $wordsim_eng $freq_eng >> puk_d300_nm_l2n.tsv
python3.7 src/floating_cent_newmes_nol2n.py $puk $wordsim_eng $freq_eng >> puk_d300_nm_nol2n.tsv

for i in 0 1 3 
do
    python3.7 src/logfreq_scalpro_l2n.py $sde $wordsim_de $freq_de $i >> sde_d300_lsa_l2n_$i.tsv
    python3.7 src/logfreq_scalpro_nol2n.py $puk $wordsim_eng $freq_eng $i >> sde_d300_lsa_l2n_$i.tsv
done
