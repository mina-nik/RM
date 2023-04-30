

import redis 
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from os.path import join as path_join
from pathlib import Path
import time
from numpy import random, save
import os


def conver_performance_to_redis(file_name: str, db_index: int = 0):
    """
    it gets a performance file (file_name) and converts it to redis db format and save it to db_num

    Parameters:
    -----------
    file_name: str
        Path to the csv file containing the performance information.
    db_index: int
        The index of the redis database to save the performance information to.    

    Output:
    -------
    A redis database with the performance information (db_index).

    """
    measurements_csv = pd.read_csv(file_name) 
        
    
    if not isinstance(measurements_csv, pd.core.frame.DataFrame):
        raise ValueError("Input type for workload should be pd.core.frame.DataFrame!")
    
    tmp_dic_measurement = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))) 
    
    for i in range(len(measurements_csv)):
        job_name = measurements_csv.loc[i]['job_name']
        proc = measurements_csv.loc[i]['proc']
        

        tmp_dic_measurement[job_name][proc]  = {'energy': measurements_csv.loc[i]['energy'], 'exe_time': measurements_csv.loc[i]['exe_time'], 'power': measurements_csv.loc[i]['energy'] / measurements_csv.loc[i]['exe_time']}



    #######db_index is used for saving measurement in redis datasets#################
    #############################################################################
    r_out = redis.Redis(db = db_index)
    r_out.flushdb()
    

    ###########adding the values as string to dataset############################
    with r_out.pipeline() as pipe:
        for key, value in tmp_dic_measurement.items():
            pipe.mset({str(key): json.dumps(dict(value))}) #converts the dictionary to string (s = json.dumps(value)), the inverse convert is variables2=json.loads(s), dict(value) to convert defaultdict to dict
        
        pipe.execute()
        
    
    saved = False
    while not saved:
        try:
            r_out.bgsave()
            saved = True
        except:
            time.sleep(db_index*2)

    
   



def migration_cost_to_redis(processors: list, coeff: float = 0.2, db_index: int = 0):
    """
    based on the redis performace database creates two redis dbs for migration overhead in terms of time and energy.

    parameters:
    -----------
    processors: list
        list of processors in the system.
    coeff: float
        the coefficient for migration overhead.
    db_index: int
        the index of the redis database to read performance data from.

    Output:
    -------
    two redis databases for migration overhead in terms of time (db_index+1) and energy (db_index+2).

    """
    r_in = redis.Redis(db = db_index)
    r_out_t = redis.Redis(db = (db_index+1))
    r_out_e = redis.Redis(db = (db_index+2))
    r_out_t.flushdb()
    r_out_e.flushdb()

    tmp_dic_migration_e = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    tmp_dic_migration_t = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

  
    n = len(processors)
    for job_name in r_in.keys():
        val = r_in.get(job_name)
        dic_val = json.loads(val)
        e = 0
        t = 0
        for _, val in dic_val.items():
            e += val['energy']
            t += val['exe_time']
            for p1 in processors:
                for p2 in processors:
                    if p1 == p2:
                        tmp_dic_migration_e[job_name][p1][p2] = 0
                        tmp_dic_migration_t[job_name][p1][p2] = 0
                    else:
                        tmp_dic_migration_e[job_name][p1][p2] = coeff * e/n 
                        tmp_dic_migration_t[job_name][p1][p2] = coeff * t/n 


    ########### writing overhead vlues to redis db ############################
    time.sleep(1)

    # write energy migration overhead to redis
    with r_out_e.pipeline() as pipe:
        for key, value in tmp_dic_migration_e.items():
            pipe.mset({key: json.dumps(dict(value))}) 
        pipe.execute()
    
    savedm = False
    while not savedm:
        try:
            r_out_e.bgsave()
            savedm = True
        except:
            time.sleep(db_index*2)


    # write time migration overhead to redis
    with r_out_t.pipeline() as pipe:
        for key, value in tmp_dic_migration_t.items():
            pipe.mset({key: json.dumps(dict(value))}) 
        pipe.execute()
    
    savedm1 = False
    while not savedm1:
        try:
            r_out_t.bgsave()

            savedm1 = True
        except:
            time.sleep(db_index*2)

