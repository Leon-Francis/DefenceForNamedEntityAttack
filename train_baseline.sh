currTime=$(date +"%Y-%m-%d_%T")

nohup python ./baseline/baseline_train.py > ./baseline/logs/IMDB_LSTM_${currTime}_0.log 2>&1 &