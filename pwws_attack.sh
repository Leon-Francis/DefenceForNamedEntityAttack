currTime=$(date +"%Y-%m-%d_%T")

nohup python ./pwws/pwws_attack.py > ./pwws/logs/PWWS_attack_IMDB_Bert_attach_NE_only_NE_attack_${currTime}_0.log 2>&1 &