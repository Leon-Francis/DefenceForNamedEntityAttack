currTime=$(date +"%Y-%m-%d_%T")

nohup python ./baseline/baseline_train.py > ./baseline/logs/IMDB_Bert_adversial_training_${currTime}_0.log 2>&1 &