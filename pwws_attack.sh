currTime=$(date +"%Y-%m-%d_%T")

nohup python ./pwws/pwws_attack.py > ./pwws/logs/AGNEWS_Bert_attach_NE_${currTime}_1.log 2>&1 &