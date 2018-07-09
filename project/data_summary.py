if __name__ == '__main__':
    import os
    from collections import Counter

    c=Counter()
    for dir in ['300VW', '300W', 'afw', 'helen', 'ibug', 'lfpw']:
        for _, _, filenames in os.walk(os.path.join( '../data',dir)):
           for filename in filenames:
              c.update([filename.split('.')[-1]])

    print('data postfix : %s'%c)
