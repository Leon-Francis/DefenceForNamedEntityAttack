currTime=$(date +"%Y-%m-%d_%T")

nohup python ./baseline/baseline_train.py > ./baseline/logs/AGNEWS_Bert_attach_NE_${currTime}_0.log 2>&1 &