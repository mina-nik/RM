
import pandas as pd
from collections import defaultdict
import re
import random
import redis

    
    
class HardwareSpecification(object):
    """
     Hardware specification of the system.
     
    """
    def __init__(self, performace_file, n_cpu , n_gpu ,db_index):
        """
        Constructor of the HardwareSpecification class.

        Parameters:
        -----------
            performace_file: str
                Path to the csv file containing the performance information.
            n_cpu: int
                Number of CPUs.
            n_gpu: int
                Number of GPUs.
        """
        # Hardware specifications
        self.n_cpu = n_cpu
        self.n_gpu = n_gpu
        self.cpu_gpu_remained_capacity = {}
        self.static_power_consumption = 0
        
        # ID creation of processing elements
        self.processors = []
        for i in range(1,self.n_cpu+1):
            self.processors.append('cpu'+str(i))
        for i in range(1,self.n_gpu+1):
            self.processors.append('gpu'+str(i))
       
        
        self.measurement = redis.Redis(db = (db_index *3))
        self.t_migration = redis.Redis(db = (db_index *3 +1)) 
        self.e_migration = redis.Redis(db = (db_index *3 +2))

        
        
                    
                    
                    
