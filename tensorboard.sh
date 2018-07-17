(nohup tensorboard --port 10086 --logdir=logs/logs_5_blocks > logs/tensorboard.out 2>&1 &) || (tail -f logs/tensorboard.out)
