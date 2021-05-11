currTime=$(date +"%Y-%m-%d_%T")

nohup python ./baseline/baseline_train.py > ./baseline/logs/train_baseline_IMDB_Bert_${currTime}_0.log 2>&1 &