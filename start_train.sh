(nohup python3 main.py -phase=train >logs/train.out 2>&1 &) || (tail -f logs/train.out)
