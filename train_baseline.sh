currTime=$(date +"%Y-%m-%d_%T")

nohup python ./baseline/baseline_train.py > ./baseline/logs/IMDB_TextCNN_baseline_${currTime}_1.log 2>&1 &