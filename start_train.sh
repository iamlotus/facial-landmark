#!/bin/sh

# multiple command in one line
nohup python3 face_landmark.py -mode=train >logs/train.out 2>&1 & echo $! > .trainpid && tail -f logs/train.out
