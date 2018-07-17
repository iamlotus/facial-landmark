import threading
import time


count=120

def fun_timer():
    print(time.time())
    global timer
    global count

    count-=1
    if count>0:
        timer = threading.Timer(1,fun_timer)
        timer.start()

timer = threading.Timer(1,fun_timer)
timer.start()