#!/bin/sh

# multiple command in one line

if [ -f ".trainpid" ]; then
  if [ -d /proc/`cat .trainpid` ]; then
     echo found running pid `cat .trainpid`
  else
     echo [remove dead pid `cat .trainpid`] && rm .trainpid
     echo [start training ...] && nohup python3 face_landmark.py -mode=train >logs/train.out 2>&1 & echo $! > .trainpid && tail -f logs/train.out
  fi
else
  echo [start training ...] && nohup python3 face_landmark.py -mode=train >logs/train.out 2>&1 & echo $! > .trainpid && tail -f logs/train.out
fi