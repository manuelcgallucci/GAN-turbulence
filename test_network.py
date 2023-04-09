import os
import time


start = time.time()

with open("zLog.txt", "w") as f:
        f.write("My process pid is: {:d}".format(os.getpid()))

loops = 1500
k = 0
while k < loops:
    current_time = time.strftime("%H:%M:%S", time.localtime())
    with open("zLog.txt", "a+") as f:
        f.write("\nCurrent time is: {:s}. I have been awake [s]: {:f}".format(current_time, time.time() - start))
    k += 1
    time.sleep(30)

with open("zLog.txt", "a+") as f:
        f.write("finished")