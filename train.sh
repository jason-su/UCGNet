LOGFILE=/sujh/logs/split-`date +%Y-%m-%d-%H-%M-%S`.log
CUDA_VISIBLE_DEVICES=0,1 nohup python -u train.py --iter=100000 >>$LOGFILE 2>&1 &