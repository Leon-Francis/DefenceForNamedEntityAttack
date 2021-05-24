currTime=$(date +"%Y-%m-%d_%T")

nohup python ./baseline/baseline_train.py > ./baseline/logs/SST2_Bert_attach_NE_${currTime}_1.log 2>&1 &