#!/bin/sh

if [ -n "$1" ]; then
    python3 face_landmark.py -cuda_visible_devices=2 -mode=eval -network="$1"
else
    echo "Usage: start_train.sh [mode], mode can be cnn/rnn"
fi