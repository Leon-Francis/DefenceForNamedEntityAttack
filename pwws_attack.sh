currTime=$(date +"%Y-%m-%d_%T")

nohup python ./pwws/pwws_attack.py > ./pwws/logs/PWWS_attack_IMDB_Bert_MNE_${currTime}_1.log 2>&1 &