#!/bin/sh

# multiple command in one line

if [ -f ".trainpid" ]; then
  echo found running instance `cat .trainpid`
else
  echo [start training ...]
  nohup python3 face_landmark.py -mode=train >logs/train.out 2>&1 & echo $! > .trainpid && tail -f logs/train.out
fi