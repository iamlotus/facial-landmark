#!/bin/sh

# multiple command in one line

if [ -n "$1" ]; then
    if [ -f ".trainpid" ]; then
        if [ -d /proc/`cat .trainpid` ]; then
            echo found running pid `cat .trainpid`
        else
            rm .trainpid \
            && echo [remove dead pid `cat .trainpid`] \
            && nohup python3 face_landmark.py -mode=train -network="$1" -save_checkpoints_secs=1200 >logs/train_"$1".out 2>&1 & echo $! > .trainpid \
            && echo [train "$1" started] \
            && busybox tail -f logs/train_"$1".out
        fi
    else
        nohup python3 face_landmark.py -mode=train -network="$1" -save_checkpoints_secs=1200 >logs/train_"$1".out 2>&1 & echo $! > .trainpid \
        && echo [train "$1" started] \
        && busybox tail -f logs/train_"$1".out
    fi
else
    echo "Usage: start_train.sh [mode], mode can be cnn/rnn"
fi


