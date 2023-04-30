
import pulp
import pandas as pd
import re
from collections import defaultdict
import numpy as np
import json
from gurobipy import *
from os.path import join as path_join
import resource_consumption as rp
from itertools import combinations
from itertools import permutations
import queue
import utils as axf
import hw_specification as hw



    
class Heuristic(object):

    """
    Heuristic class for solving mapping/scheduling problem.
    """

    def __init__(self, current_state_file: str, performace_file: str, current_time: float, n_cpu: int, n_gpu: int, db_index: int = 0):
       
       """
        Constructor for the Heuristic class.

        Parameters:
        -----------
            current_state_file: str
                Path to the csv file containing the current state information.
            performance_file: str
                Path to the csv file containing the performance information.
            current_time: float
                The current time in seconds.
            n_cpu: int
                The number of CPUs available.
            n_gpu: int
                The number of GPUs available.
            db_index: int, optional (default=0)
                The index of the database to use.

        """
       
       self.current_state = pd.read_csv(current_state_file)  # read the current state csv file
       self.performance = pd.read_csv(performace_file) # read the performance csv file
       self.current_time = current_time #float
       self.hw = hw.HardwareSpecification(performace_file, n_cpu, n_gpu,db_index) # create a HardwareSpecification object
       self.rp = rp.ResourceConsumption(self.hw) # create a ResourceConsumption object
       self.f_i = defaultdict(lambda: defaultdict(lambda: -1000000)) #  create a defaultdict object to save energy cost values


    def d_calculation(self, current_time, key, job_name, deadline, curr_proc, progress, arrival_time, is_pred, t_mig, e_mig, overhead):
        """
        Calculate the d value (difference between the second smallest and the smallest desirability) for a job.

        Parameters
        ----------
            current_time : float
                The current time.
            key : int
                The key for the job.
            job_name : str
                The name of the job.
            deadline : float
                The deadline for the job.
            curr_proc : str
                The current processor for the job.
            progress : float
                The progress of the job.
            arrival_time : float
                The arrival time of the job.
            is_pred : bool (1 or 0)
                A flag indicating whether the job is a prediction job.
            t_mig : float
                The migration time for the job.
            e_mig : float
                The energy consumption for migration of the job.
            overhead : float
                The overhead.

        Returns
        -------
            float
                The calculated d value for the job.
    """

        
        C = -100**100
        
        left_time = deadline - max(current_time, arrival_time) 
        f = [(proc,self.rp.get_energy_consumption(key, job_name, curr_proc, progress, proc, is_pred, e_mig) + np.float128(C) * np.float128(int(self.rp.get_execution_time(key, job_name, curr_proc, progress, proc, is_pred, t_mig, 0) > left_time))) for proc in self.hw.processors if self.rp.get_execution_time(key, job_name, curr_proc, progress, proc,is_pred, t_mig, 0) <= self.hw.cpu_gpu_remained_capacity[proc]] #Fi
        self.f_i[key] = {k:v for k,v in f if v > 0}
        
        if not len(self.f_i[key]):
            d = np.nan
        
        else:
        
            proc_star = min(self.f_i[key], key=self.f_i[key].get) # the best processor for job i
            proc_star_energy = self.f_i[key][proc_star]
            self.f_i[key].pop(proc_star)
            
            if not len(self.f_i[key]):
                d = float("inf")
            else:
                proc_prime = min(self.f_i[key], key=self.f_i[key].get) # the second best processor for job i
                d = self.f_i[key][proc_prime] - proc_star_energy # d is the difference between the second best processor and the best processor for job i
            self.f_i[key][proc_star] = proc_star_energy
        
        return d   
        
    
    def is_schedulable(self, current_y, row_star, proc_star, nonpreemptable_jobs, write_schedule):
       
        """
        Checks the schedulability of jobs corresponds to row_star on resource proc_star at currenttime given the set of tasks that are mapped on proc_star so far.

        Parameters
        ----------
            current_y : dict
                The current mapping vector.
            row_star : Pandas Series object
                Row of the workload that corresponds to the job to be scheduled.
            proc_star : str
                The processor to be scheduled on.
            nonpreemptable_jobs : dict
                Dictionary of nonpreemptable jobs.
        

        Returns
        -------
            True if it is schedulable, otherwise False.
    """

        ###### initialization ########
        job_star = row_star['key']
        job_name = row_star['job_name']
        curr_proc = row_star['proc'] 


        remaining_exe_time = self.rp.execution_time(row_star, proc_star)
        if self.hw.cpu_gpu_remained_capacity[proc_star] < remaining_exe_time: # there is no enough capacity on the processor
            return False
        

        mapped_jobs_on_proc_star_keys = [k for k,v in current_y.items() if str(v) == proc_star] + [job_star]
        mapped_jobs_on_proc_star_df = (self.rp.all_jobs_df.loc[self.rp.all_jobs_df['key'].isin(mapped_jobs_on_proc_star_keys)].sort_values(by='deadline').reset_index(drop=True)) #new


        if axf.strip_digits(proc_star) == "gpu" and nonpreemptable_jobs[proc_star] in mapped_jobs_on_proc_star_keys:
            without_stop = True # if the job can be scheduled without stopping for non-preemptable job
        else:
            without_stop = False

        iteration_counter = 1 # to count the number of iterations in the main while loop


        # If write_schedule is True, then the schedule will be written in the schedule file
        if write_schedule:
            schedule_tmp = pd.DataFrame(columns = ['key','arrival_time','deadline','job_name','proc','start','end'])
            

        ###### end of initialization ########

        while iteration_counter <= 2:

            if iteration_counter > 1 and schedulable: # a schedulable schedule is perviously found
                break


            schedulable = True
            next_scheduling_point = 0
            tmp_job_df = mapped_jobs_on_proc_star_df.copy()


            if without_stop and iteration_counter == 1: # check if the job can be scheduled without stopping for non-preemptable job
                
                non_preemptive_job_on_proc_star = mapped_jobs_on_proc_star_df[mapped_jobs_on_proc_star_df['key'] == nonpreemptable_jobs[proc_star]].squeeze()
                next_scheduling_point = self.rp.execution_time(non_preemptive_job_on_proc_star, proc_star)
                if next_scheduling_point > (non_preemptive_job_on_proc_star['deadline'] - self.current_time):
                    schedulable = False
                    if write_schedule:
                        schedule_tmp = pd.DataFrame(columns = ['key','arrival_time','deadline','job_name','proc','start','end'])
                    iteration_counter += 1
                    continue 
                
                tmp_job_df = tmp_job_df[tmp_job_df.key != nonpreemptable_jobs[proc_star]] 
                if write_schedule:
                    tmp_d = pd.DataFrame({'key':[non_preemptive_job_on_proc_star['key']],'arrival_time': [non_preemptive_job_on_proc_star['arrival_time']],'deadline': [non_preemptive_job_on_proc_star['deadline']],'job_name':[non_preemptive_job_on_proc_star['job_name']],'proc': [proc_star],'start': [0+self.current_time],'end': [next_scheduling_point+self.current_time]})  
                    schedule_tmp = pd.concat([schedule_tmp, tmp_d], ignore_index=True)

           
            if len(tmp_job_df) == 0: # there is no job remianed to be scheduled
                break

            tmp_job_df.sort_values(by='deadline', inplace=True, ignore_index=True) # EDF order
            SU_df = pd.DataFrame(columns= tmp_job_df.columns) # suspended jobs dataframe
            SU_df['delayed'] = 0 # to indicate if the job is delayed or not
            gap = (max(mapped_jobs_on_proc_star_df['deadline']) - self.current_time) - next_scheduling_point # jobs shoyld be scheduled in this gap
            iteration_counter += 1
            
            
            while (len(tmp_job_df) or len(SU_df)) and schedulable: 
                
                remained_portion = 1

                if len(tmp_job_df):
                    shortest_deadline_job_index = tmp_job_df.index[0] # index of the job with the earliest deadline
                    j_prime = tmp_job_df.loc[shortest_deadline_job_index]['key'] # key of the job with the earliest deadline
                    j_prime_prime = tmp_job_df.loc[shortest_deadline_job_index]['key'] # copy of the key of the job with the earliest deadline
                    
                
                j_prime_processed = False

                while (not j_prime_processed) and schedulable:

                    if len(SU_df): # check if there is arrived job in SU

                        if not len(tmp_job_df):

                            j_prime_processed = True

                            time_gap =  0
                            if (SU_df['delayed'] == 0).any():
                                min_arrival_time = SU_df.query("delayed == 0")['arrival_time'].min()  
                                if not len(tmp_job_df):
                                    time_gap =  (min(SU_df[SU_df['delayed'] == 0]['arrival_time']) -self.current_time) - next_scheduling_point
                            else:
                                if not len(tmp_job_df):
                                    SU_df['delayed'] = 0
                                    min_deadline = min(SU_df['deadline'])
                                    time_gap =  (min(SU_df[SU_df['deadline'] == min_deadline]['arrival_time']) - self.current_time) - next_scheduling_point 
                                    min_arrival_time = min(SU_df[SU_df['deadline'] == min_deadline]['arrival_time'])

                                else:
                                    min_arrival_time = float("inf")

                            if time_gap > 0:
                                next_scheduling_point += time_gap
                                
                               
                            # next_scheduling_point might be needed to update untill arrival of the first job in SU if needed

                        else:
                            if (SU_df['delayed'] == 0).any():
                                min_arrival_time = SU_df.query("delayed == 0")['arrival_time'].min()  
                                
                            else:
                                min_arrival_time = float("inf")

                        
                        # choose the job with the earliest arrival from SU which is aleredy arrived
                        row_min = SU_df.loc[SU_df['arrival_time'] == min_arrival_time].squeeze()

                        # a spacial case for gpu since jobs cannot be broken to multiple chunks
                        if min_arrival_time == self.current_time and axf.strip_digits(proc_star) == "gpu":
                                     
                                     deadline_of_job_with_earliest_arrival = SU_df.loc[SU_df['arrival_time'] == self.current_time]['deadline'].values[0]
                                     higher_priority_jobs_df = SU_df[SU_df['deadline'] <  deadline_of_job_with_earliest_arrival]

                                     if len(higher_priority_jobs_df):
                                         arrival_of_job_with_higher_priority = min(higher_priority_jobs_df['arrival_time']) - self.current_time
                                         gap_tmp = arrival_of_job_with_higher_priority - self.current_time - next_scheduling_point 
                                         if gap_tmp < self.rp.execution_time(mapped_jobs_on_proc_star_df.loc[mapped_jobs_on_proc_star_df['arrival_time'] == self.current_time]['key'].values[0], proc_star):
                                             SU_df.loc[SU_df['arrival_time'] == self.current_time, 'delayed'] = 1 # delay the job
                                             min_arrival_time = min(SU_df[SU_df['delayed'] == 0]['arrival_time'])
                                             row_min = mapped_jobs_on_proc_star_df.loc[mapped_jobs_on_proc_star_df['arrival_time'] == min_arrival_time].squeeze()

                                             if not len(tmp_job_df):
                                                 if (SU_df['delayed'] == 0).any():
                                                    min_arrival_time = min(SU_df[SU_df['delayed'] == 0]['arrival_time'])
                                                    row_min = SU_df.loc[SU_df['arrival_time'] == min_arrival_time].squeeze()
                                                 else:
                                                    min_arrival_time = min(SU_df[SU_df['arrival_time'] != self.current_time]['arrival_time'])
                                                    row_min = SU_df.loc[SU_df['arrival_time'] == min_arrival_time].squeeze()



                        if min_arrival_time - self.current_time <= next_scheduling_point:
                            SU_df['delayed'] = 0
                            j_prime_prime = row_min['key']
                            remained_portion = row_min['remanied_portion']
                            row_min = SU_df.loc[SU_df['arrival_time'] == min_arrival_time].squeeze()
                            SU_df = SU_df[SU_df.key != j_prime_prime]

                            # find highest priority job in SU_df
                            higher_priority_jobs_df = SU_df[SU_df['deadline'] <  row_min['deadline']]
                            if len(higher_priority_jobs_df):
                                gap = min(higher_priority_jobs_df['arrival_time']) - self.current_time - next_scheduling_point # gap should be updated until arrival of the highest priority job
                            else:
                                gap = max(mapped_jobs_on_proc_star_df['deadline']) - self.current_time - next_scheduling_point
                        
                        

                    if j_prime == j_prime_prime and j_prime_processed == False:
                        j_prime_processed = True
                        tmp_job_df = tmp_job_df.drop(shortest_deadline_job_index) # fully new part!!!!

                    selected_job = mapped_jobs_on_proc_star_df[mapped_jobs_on_proc_star_df['key'] == j_prime_prime].squeeze()

                    if selected_job['arrival_time'] > self.current_time + next_scheduling_point: # job is not arrived yet

                        # add it to SU_df
                        selected_job['remanied_portion'] = 1
                        selected_job['delayed'] = 1     
                        SU_df = pd.concat([SU_df,selected_job.to_frame().transpose()], ignore_index = True)
                        
                        # update gap
                        gap = (min(SU_df['arrival_time']) - self.current_time) - next_scheduling_point 
                        continue

                        
                    else: # job is arrived


                        if not len(tmp_job_df) and (j_prime != j_prime_prime):
    
                            selected_job = row_min 
                            key = selected_job['key']
                            remained_portion = selected_job['remanied_portion']
                            deadline = selected_job['deadline']
                            SU_df = SU_df[SU_df.key != key]
                            if len(SU_df) > 0:
                                higher_priority_jobs = SU_df[SU_df['deadline'] <  deadline]
                            else:
                                higher_priority_jobs = pd.DataFrame()
                            if len(higher_priority_jobs):
                                a_p2 = min(higher_priority_jobs['arrival_time']) - self.current_time
                                gap = a_p2 - next_scheduling_point
                            #else:
                                #gap = max(mapped_jobs_on_proc_star_df['deadline']) - self.current_time - next_scheduling_point

                            time_gap =  (selected_job['arrival_time'] - self.current_time) - next_scheduling_point
                  
                            if time_gap > 0:
                                exe_time_sum += time_gap


                        
                        if selected_job['key'] == nonpreemptable_jobs[proc_star]: #
                            if next_scheduling_point == 0:  # without preepmtion
                                exe_time = self.rp.execution_time(selected_job, proc_star) * remained_portion
                            else: # with preemption
                                exe_time = self.rp.execution_time(selected_job, proc_star,'reset') 
                        else:
                            exe_time = self.rp.execution_time(selected_job, proc_star) * remained_portion

                        # prilimanary deadline check
                        if next_scheduling_point + exe_time > (selected_job['deadline'] - self.current_time):
                            schedulable = False
                            if write_schedule:
                                schedule_tmp = pd.DataFrame(columns = ['key','arrival_time','deadline','job_name','proc','start','end'])
                            break
                        
                        if exe_time <= gap :
                            selected_job['start'] = next_scheduling_point
                            next_scheduling_point += exe_time 
                            selected_job['end'] =  next_scheduling_point
                            

                            
                        else: # exe_time > gap
                            
                            portion = 0
                            delayed = 1
                            if axf.strip_digits(proc_star) == "cpu":
                                portion = gap / exe_time 
                                

                            row_tmp = selected_job.copy()
                            selected_job['start'] = next_scheduling_point
                            next_scheduling_point += portion * (exe_time * remained_portion)

                            if portion < 1:
                                row_tmp['delayed'] = delayed
                                row_tmp['remanied_portion'] = remained_portion * (1 - portion)
                                SU_df = pd.concat([SU_df,row_tmp.to_frame().transpose()], ignore_index = True)

                            selected_job['end'] = next_scheduling_point

                        
                        if write_schedule:
                            tmp_d = pd.DataFrame({'key':[selected_job['key']],'arrival_time':[selected_job['arrival_time']],'deadline':[selected_job['deadline']],'job_name':[selected_job['job_name']],'proc': [proc_star],'start': [selected_job['start']+self.current_time],'end': [selected_job['end']+self.current_time]})  
                            schedule_tmp = pd.concat([schedule_tmp, tmp_d], ignore_index=True)

                        # deadline check
                        if next_scheduling_point > (selected_job['deadline'] - self.current_time):
                            schedulable = False
                            if write_schedule:
                                schedule_tmp = pd.DataFrame(columns = ['key','arrival_time','deadline','job_name','proc','start','end'])
                            break 


            # end of while loop (k < len_jobs and k >= 0) or len(SU)

            if not without_stop: # in this case just one iteration is enough
                break


        # end of while loop (iteration_counter < 2)

        # if schedulable and write_schedule:
        if schedulable and write_schedule:
            base_schedule = pd.read_csv('schedule.csv')
            base_schedule = base_schedule[base_schedule['proc'] != proc_star]
            base_schedule = pd.concat([base_schedule,schedule_tmp], ignore_index=True)
            base_schedule.to_csv('schedule.csv', index = False)
            


        return schedulable


            






    def solve(self, write_schedule: bool = False):
       """
        Solves the mapping/scheduling problem using heuristic approach.

        Parameters:
        -----------
            write_schedule: bool, optional (default=False)
                Whether to write the schedule to schedule.csv.


        Returns:
        --------
        srt
            'Optimal' if a feasible solution is found, otherwise returns 'Infeasible'.

        """
       ################## itialization ##################
       #################################################
       
       self.current_state['prev_proc'] = self.current_state['proc']

       # Predictions are parts of current_state where arrival_time is greater than current_time
       prediction_df = self.current_state[self.current_state['arrival_time'] > self.current_time].copy()
       
       # current_mode is part of current_state that their arrival time is less than or equal to current_time
       current_mode_df = self.current_state[self.current_state['arrival_time'] <= self.current_time].copy()

       # If there are no rows in current_mode_df, do not run ILP, only for prediction  
       if not len(current_mode_df): 
           return False 
       
       current_mode_df_init = current_mode_df.copy()
       prediction_df_init = prediction_df.copy()
       

       # Data type conversion of columns in current_mode_df
       current_mode_df[['proc', 'prev_proc']] = current_mode_df[['proc', 'prev_proc']].astype(str)
       current_mode_df['is_pred'] = 0
       
       
       # Separate non-preemptable jobs for non-preemptable GPUs
       current_mode_nonpreemptable = current_mode_df.query("proc.str.contains('gpu') and is_pred != 1 and progress > 0").reset_index(drop=True) 
       nonpreemptable_jobs_keys = current_mode_nonpreemptable['key'].tolist()

       # Extract non-preemptable job GPUs
       nonpreemptable_jobs = defaultdict(lambda: -1)
       if len(current_mode_nonpreemptable):
           for index, row in current_mode_nonpreemptable.iterrows():
              nonpreemptable_jobs[row['proc']] = row['key']
       

       # If wirte_schedule is True, then write the schedule in a file
       if write_schedule:
           schedule = pd.DataFrame(columns = ['key','arrival_time','deadline','job_name','proc','start','end'])
           schedule.to_csv('schedule.csv')

       # Initialize status of MILP solver and prediction_intervene flag
       status = False 

       ####### end of initialization ####################
       #################################################
       

       while not status: 
           
           self.rp.all_jobs_df = current_mode_df.copy()
           self.rp.all_jobs_df = pd.concat([self.rp.all_jobs_df,prediction_df.copy()], ignore_index = True)

           max_capacity = max(set(self.rp.all_jobs_df['deadline'])) - (self.current_time) #max_capacity of the system is max time left for the jobs in the system

           # capacity initialization 
           self.hw.cpu_gpu_remained_capacity = defaultdict(lambda: max_capacity) 

           U = set(self.rp.all_jobs_df['key'].unique()) 
           y = defaultdict(lambda: None) # y[j] = i if job j is mapped on processor i, y is mapping vecotor

           if len(prediction_df):
               
               # Data type conversion of columns in prediction_df
               prediction_df[['proc', 'prev_proc']] = prediction_df[['proc', 'prev_proc']].astype(str)
               prediction_df['is_pred'] = 1

               prediction_df.reset_index(drop=True, inplace=True)
               pred_jobs = prediction_df['key'].unique().tolist()


           tmp_df = self.rp.all_jobs_df.copy()    
           status = True #"" 
           while U and status:
               
               # Step 1: find the optimal job to map

               # calculate d for all jobs 
               tmp_df['d'] = tmp_df.apply(lambda x: self.d_calculation(self.current_time, x['key'], x['job_name'], x['deadline'], x['proc'], x['progress'], x['arrival_time'], x['is_pred'], x['t_mig'], x['e_mig'], 0), axis=1)  

               if tmp_df['d'].isnull().values.any(): # there is a job that cannot be mapped
                    job_star = None
                    status = False 
                    U = set()
                    break
               
               else:
                   max_d = tmp_df['d'].max()
                   row_star = tmp_df[tmp_df['d'] == max_d].iloc[0].squeeze()
                   job_star = row_star['key']
                   
               # Step 2: map and shcedule the selected jon in step 1
               if job_star != None:
                    left_time = row_star['deadline'] - self.current_time
                    f_i_star = self.f_i[job_star]
                    job_name = row_star['job_name'] 
                    proc_star = ""

                    # find the best processor for job_star
                    while not y[job_star]:

                        if len(f_i_star):

                            min_val = min(f_i_star.values())
                            proc_star = [k for k,v in f_i_star.items() if f_i_star[k] == min_val][0]

                            if self.is_schedulable(y.copy(), row_star, proc_star, nonpreemptable_jobs, write_schedule):
                                y[job_star] = proc_star
                                U.remove(job_star)  
                                tmp_df = tmp_df[tmp_df['key'] != job_star].reset_index(drop=True)
                                self.hw.cpu_gpu_remained_capacity[proc_star] -= self.rp.execution_time(row_star, proc_star)

                            else: #is not schedulable
                                if len(f_i_star):
                                    f_i_star.pop(proc_star)
                        else:
                            status = False 
                            break

                    # end of while not y[job_star]
           
           # end of while U

           if not status: 
               if len(prediction_df): # drop prediction 
                  prediction_df = pd.DataFrame(columns = self.rp.all_jobs_df.columns)

               else:
                  status = False
                  break


                     
       ##end of while loop (while status != "Optimal")

       
       # if wirte_schedule is True, then write the schedule in a file 
       if write_schedule:
           if status: 
                self.rp.all_jobs_df['prev_proc'] = self.rp.all_jobs_df['proc']
                schedule = pd.read_csv('schedule.csv')
                schedule = axf.adjust(schedule)
                schedule.to_csv('schedule.csv', index=False)
           else:
                schedule = pd.DataFrame(columns = ['key','job_name','proc','start','end'])
                schedule.to_csv('schedule.csv', index=False)

       
       return "Optimal" if status  else "Infeasible"
       
     
       
      