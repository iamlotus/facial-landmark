(nohup python3 -u main.py -phase=train -num_residual_blocks=20 >logs/train.out 2>&1 &) || (tail -f logs/train.out)
