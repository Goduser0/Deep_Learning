import time
import numpy as np
class Timer(object):
    """Create a Timer, Record the time gap"""
    def __init__(self):
        self.times=[]
        self.start()
        
    def start(self):
        # record the timestamp of start
        self.tik = time.time()
        
    def stop(self):
        # record the time gap from start to stop
        self.times.append(time.time() - self.tik)
        
    def avg(self):
        # 求平均值
        return sum(self.times) / len(self.times)
    
    def sum(self):
        # 求和
        return sum(self.times)
    
    def cumsum(self):
        # 求累积和
        return np.array(self.times).cumsum().tolist()
    
timer = Timer()
timer.stop()
timer.stop()
print(timer.times[0], timer.times[1])