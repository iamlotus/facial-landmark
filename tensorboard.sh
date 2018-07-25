#!/bin/sh

if [ -n "$1" ]; then
   nohup tensorboard --port 10086 --logdir=logs/train_"$1" > logs/tensorboard.out 2>&1 &
else
   echo "Usage: tensorboard.sh [mode], mode can be cnn/rnn"
fi



