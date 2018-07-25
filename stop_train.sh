#!/bin/sh

if [ -f ".trainpid" ]; then
    echo stop `cat .trainpid` && kill `cat .trainpid` && rm .trainpid
else
    echo nothing to stop, can not find .trainpid file
fi