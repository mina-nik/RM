
import pandas as pd
from collections import defaultdict
import re
import random
import dask.dataframe as ddf
import redis
import json
import utils as axf




    
class ResourceConsumption(object):
    """
        Resource consumption of the system.
    """
    def __init__(self, hw, non_start_mig_ignore = True):
        """
        Constructor of the ResourceConsumption class.

        Parameters:
        -----------
            hw: Hardware (HardwareSpecification)
                The hardware object.
            non_start_mig_ignore: bool
                If True, the migration cost of the jobs that are not started yet is ignored.
            
        """
       
        self.hw = hw
        self.all_jobs_df = pd.DataFrame()
        self.non_start_mig_ignore = non_start_mig_ignore

    def energy_consumption(self ,row, proc, flag = '', for_write = False):
        """
        Calculates the energy consumption of the job.
        
        Parameters:
        -----------
            row: pandas.Series
                The row of the job.
            proc: str
                The processor that the job is assigned to.
            flag: str   
                If flag is 'reset', the energy consumption is calculated considering 0 progress so far.
                If flag is 'check_migrated', migration in case of GPU should be checked.
            for_write: bool
                If True, the energy consumption is calculated for the write overhead.

        Returns:
        --------
            energy consumption: float
        """
         
        
        if row.key not in self.all_jobs_df['key'].values:
            return 0

        if for_write:
            curr_proc = row['prev_proc']
        else:
            curr_proc = row['proc']

        return self.get_energy_consumption(row.key, row.job_name, curr_proc, row.progress, proc, row.is_pred, row.e_mig ,flag, for_write) 
        

        
    def execution_time(self, row, proc, flag =  '',for_write = False):
        """
        Calculates the execution time of the job on proc.
        
        Parameters:
        -----------
            row: pandas.Series
                The row of the job.
            proc: str
                The processor that the job is assigned to.
            flag: str   
                If flag is 'reset', the energy consumption is calculated considering 0 progress so far.
                If flag is 'check_migrated', migration in case of GPU should be checked.
            for_write: bool
                If True, the energy consumption is calculated for the write overhead.

        Returns:
        --------
            execution time: float
        """
        
        
        
        job = row['key']
        if row['key'] not in self.all_jobs_df['key'].values:
            return 0
       

        if for_write:
            curr_proc = row['prev_proc'] 
        else:
            curr_proc = row['proc'] 

        return self.get_execution_time(row.key, row.job_name, curr_proc, row.progress, proc, row.is_pred , row.t_mig, 0,flag , for_write)
        

       

    def get_energy_consumption(self, job, job_name, curr_proc, progress, proc, is_pred, e_mig, flag = '', for_write = False):
        """
        Calculates the energy consumption of the job.

        Parameters:
        -----------
            job: str
                The job id.
            job_name: str   
                The job name.
            curr_proc: str
                The processor that the job is assigned to.
            progress: float 
                The progress of the job.
            proc: str
                The processor that the job is assigned to.
            is_pred: bool
                If True, the job is a prediction job.
            e_mig: float
                The migration energy of the job.
            flag: str
                If flag is 'reset', the energy consumption is calculated considering 0 progress so far.
                If flag is 'check_migrated', migration in case of GPU should be checked.
            for_write: bool
                If True, the energy consumption is calculated for the write overhead.

        Returns:
        --------
            energy consumption: float
        """
        
       
        if job not in self.all_jobs_df['key'].values:
            return 0
        
             
        remained_progress = 1 - progress
        if axf.strip_digits(curr_proc) == 'gpu' or axf.strip_digits(proc) == 'gpu' and curr_proc in self.hw.processors: #reseet is needed both for moving to/leaving from gpus
            if proc != curr_proc:
                remained_progress = 1
                if for_write:
                    self.all_jobs_df.loc[self.all_jobs_df['key'] == job, 'progress'] = 0
                
             
        reset_overhead_consider = False
        if flag == 'reset':
            if axf.strip_digits(curr_proc) == 'gpu':
                if proc == curr_proc:
                    reset_overhead_consider = True
                    remained_progress = 1
            
        if flag == 'check_migrated':
            if axf.strip_digits(curr_proc) == 'gpu':
                if proc == curr_proc:
                    remained_progress = 0

        val = self.hw.measurement.get(job_name)
        measurement = json.loads(val)


        ##reset overhead (only gpus)
        reset_overhead = 0   
        if reset_overhead_consider:
            reset_overhead = (measurement[curr_proc]['energy']+ e_mig ) * progress


        ##exe_energy
        exe_energy = (measurement[proc]['energy'] +e_mig) * remained_progress
        
        #migratoin overhead
        val = self.hw.e_migration.get(job_name)
        e_migration = json.loads(val)
        if curr_proc in self.hw.processors:
            migration_energy = e_migration[curr_proc][proc]
        else:
            migration_energy = 0 

        if self.non_start_mig_ignore:
            return exe_energy + (migration_energy * int(remained_progress < 1 and remained_progress > 0)) + reset_overhead
        else:
            return exe_energy + migration_energy + reset_overhead

        
       

    
    def get_execution_time(self, job, job_name, curr_proc, progress, proc, is_pred , t_mig, overhead, flag =  '', for_write = False):
        """
        Calculates the execution time of the job.

        Parameters:
        -----------
            job: int
                The job key.
            job_name: str   
                The job name.
            curr_proc: str
                The processor that the job is assigned to.
            progress: float 
                The progress of the job.
            proc: str
                The processor that the job is assigned to.
            is_pred: bool
                If True, the job is a prediction job.
            e_mig: float
                The migration energy of the job.
            flag: str
                If flag is 'reset', the energy consumption is calculated considering 0 progress so far.
                If flag is 'check_migrated', migration in case of GPU should be checked.
            for_write: bool
                If True, the energy consumption is calculated for the write overhead.

        Returns:
        --------
            execution time: float
        """
        

        #try:
        if job not in self.all_jobs_df['key'].values:
            return 0
        #except:
        #    return 0

    
        remained_progress = 1 - progress
        if axf.strip_digits(curr_proc) == 'gpu' or axf.strip_digits(proc) == 'gpu' and curr_proc in self.hw.processors: #reseet is needed both for moving to/leaving from gpus
            if proc != curr_proc:
                remained_progress = 1
                if for_write:
                    self.all_jobs_df.loc[self.all_jobs_df['key'] == job, 'progress'] = 0
                  
        
        
        if flag == 'reset':
            if axf.strip_digits(curr_proc) == 'gpu':
                if proc == curr_proc:
                    remained_progress = 1

      
        
        #exe_time
        val = self.hw.measurement.get(job_name)
        measurement = json.loads(val)
        exe_time = (measurement[proc]['exe_time'] + t_mig + overhead) * remained_progress 

        
        val = self.hw.t_migration.get(job_name)
        t_migration = json.loads(val)
        if curr_proc in self.hw.processors:
            migration_time = t_migration[curr_proc][proc]
        else:
            migration_time = 0

        if self.non_start_mig_ignore:
            return exe_time + migration_time  * int(remained_progress < 1 and remained_progress > 0) 
        else:
            return exe_time + (migration_time)

       
       
        
                    
