
import pulp
import pandas as pd
from collections import defaultdict
import numpy as np
import json
from gurobipy import *
from os.path import join as path_join
import resource_consumption as rp
import queue
import utils as axf
import hw_specification as hw


    
    
class MILP(object):

    """
    MILP class for solving mapping/scheduling problem using Mixed Integer Linear Programming (MILP).
    """

    def __init__(self, current_state_file: str,performace_file: str,current_time: float, n_cpu: int,n_gpu: int, db_index: int = 0):
       
       """
        Constructor for the MILP class.

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
       self.current_time = current_time 
       self.hw = hw.HardwareSpecification(performace_file, n_cpu, n_gpu,db_index) # create a HardwareSpecification object
       self.rp = rp.ResourceConsumption(self.hw) # create a ResourceConsumption object
       self.current_mode_nonpreemptable_jobs_df = pd.DataFrame() # create a dataframe for non-preemptable jobs
        

    def solve(self, gurobi_solver: bool = False, write_schedule: bool = False):
       
       """
        Solves the optimization problem using the Mixed Integer Linear Programming (MILP) approach.

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
       if not len(current_mode_df): # do not want to run ILP only for prediction
           return False
       
       
       # Data type conversion of columns in current_mode_df
       current_mode_df[['proc', 'prev_proc']] = current_mode_df[['proc', 'prev_proc']].astype(str)
       current_mode_df['is_pred'] = 0

       
       # Separate non-preemptable jobs for non-preemptable GPUs
       self.current_mode_nonpreemptable_jobs_df = current_mode_df.query("proc.str.contains('gpu') and is_pred != 1 and progress > 0").reset_index(drop=True) 
       nonpreemptable_jobs_keys = self.current_mode_nonpreemptable_jobs_df['key'].tolist()

       # Extract non-preemptable jobs of GPUs
       nonpreemptable_jobs_on_proc_dict = defaultdict(lambda: -1)
       if len(self.current_mode_nonpreemptable_jobs_df):
           for _, row in self.current_mode_nonpreemptable_jobs_df.iterrows():
              nonpreemptable_jobs_on_proc_dict[row['proc']] = row['key']

       

       # Create list of GPUs
       gpu_processors = [p for p in self.hw.processors if axf.strip_digits(p) == "gpu"]
       

        # Initialize status of MILP solver 
       status = "" 

       ####### end of initialization ####################
       #################################################
       
       

       while status != "Optimal":
          
          self.rp.all_jobs_df = current_mode_df.copy()
          self.rp.all_jobs_df = pd.concat([self.rp.all_jobs_df,prediction_df.copy()], ignore_index = True)

          no_digits = axf.count_digits(max(set(self.rp.all_jobs_df['deadline'])))
          M =  10**(no_digits) # big M for linearization 


          jobs = self.rp.all_jobs_df['key'].unique().tolist()
          # Preemptable jobs
          preemptable_jobs = self.rp.all_jobs_df.loc[~self.rp.all_jobs_df['key'].isin(nonpreemptable_jobs_keys)].reset_index(drop=True)

          max_capacity = max(set(self.rp.all_jobs_df['deadline'])) - (self.current_time) # max_capacity of the system is max time left for the jobs in the system


          # Capacity initialization
          """self.hw.cpu_gpu_remained_capacity = defaultdict(lambda: max_capacity)"""

          # Sort all dataframes based on deadline
          prediction_df.sort_values(by='deadline', inplace=True, ignore_index=True)
          self.rp.all_jobs_df.sort_values(by='deadline', inplace=True, ignore_index=True)
          current_mode_df.sort_values(by='deadline', inplace=True, ignore_index=True)

          prediction_intervene = False  
          if len(prediction_df):

                # Data type conversion of columns in prediction_df
                prediction_df[['proc', 'prev_proc']] = prediction_df[['proc', 'prev_proc']].astype(str)
                prediction_df['is_pred'] = 1
                
                pred_jobs = prediction_df['key'].unique().tolist()


                # Check if the deadline of the earliest predicted job is less than the deadline of the latest current job
                if len(prediction_df):
                    prediction_deadline = set(prediction_df['deadline'])
                    current_deadline = set(current_mode_df['deadline']) if len(current_mode_df) > 0 else set()
                    prediction_intervene = min(prediction_deadline, default=0) < max(current_deadline, default=0)
            

          if prediction_intervene:
            # Select jobs with deadlines less than or equal to the earliest predicted job deadline
            priority_jobs_df = current_mode_df[current_mode_df.deadline <= min(set(prediction_df['deadline']))]
            # Select the remaining jobs after prioritizing the jobs
            remaining_jobs_df = pd.concat([current_mode_df[current_mode_df.deadline > min(set(prediction_df['deadline'])) ],prediction_df.copy()], ignore_index = True)
          else:
            # Concatenate the current mode and predicted jobs when there's no intervention
            priority_jobs_df = pd.concat([current_mode_df.copy(),prediction_df.copy()], ignore_index = True)
            remaining_jobs_df = pd.DataFrame()

          # Sort the jobs based on their deadline
          if len(priority_jobs_df):
              priority_jobs_df.sort_values(by='deadline', inplace=True, ignore_index=True)
          if len(remaining_jobs_df):
              remaining_jobs_df.sort_values(by='deadline', inplace=True, ignore_index=True)


          ####optimization part########
          #############################

          # Step 1: Define the problem
          model = pulp.LpProblem("energy_minimising_mapping_problem", pulp.LpMinimize)

          # Step 2: Define the decision variables
          x = pulp.LpVariable.dicts("x",
                                     ((j,p) for j in jobs  for p in self.hw.processors),
                                     lowBound = 0,
                                     upBound = 1,
                                     cat = pulp.LpInteger) # x[j,i] = 1 if job j is mapped to processor i, mapping variable
          
          f = pulp.LpVariable.dicts("f", (self.hw.processors), lowBound = 0, upBound = max_capacity, cat='Continuous') # f[i] is the time that the processor is free after executing nonpredicted jons on i
          a = pulp.LpVariable.dicts("a", (gpu_processors), 0, 1, pulp.LpInteger)  # a[i] = 1 if the processor is gpu and the nonpreemptive job is restarted again on this same GPU

          y = pulp.LpVariable.dicts("proc_active", (self.hw.processors), 0, 1, pulp.LpInteger) # y[i] = 1 if the processor is active

          xa = pulp.LpVariable.dicts("xa", ((p) for p in gpu_processors), lowBound = 0, upBound = 1, cat=pulp.LpInteger) # ax[i] = x[j,i]*a[i], only for gpu processors and nonpreemptable job running on it
          xa_comp = pulp.LpVariable.dicts("xa_comp", ((p) for p in gpu_processors), lowBound = 0, upBound = 1, cat=pulp.LpInteger) # xa_comp[i] = x[j,i]*(1-a[i]), only for gpu processors and nonpreemptable job running on it

          if len(prediction_df):
              or_x_preds = pulp.LpVariable.dicts("or_x_preds",(self.hw.processors), 0, 1, pulp.LpInteger) # or_x_preds[i] = 1 if there is at least one job in prediction mapped to processor i
            
              if prediction_intervene == False:
                  
                  chunk_no_pred = len(prediction_df)

                  fx = pulp.LpVariable.dicts("fx", ((j,p) for j in pred_jobs for p in self.hw.processors), lowBound = 0, upBound = max_capacity, cat='Continuous') # fx[j,i] = f[i] * x[j,i] for predicted jobs
                  s_p_j = pulp.LpVariable.dicts("s_p_j",
                            ((j,p,k) for j in pred_jobs for p in self.hw.processors for k in range(chunk_no_pred)),
                            lowBound = 0,
                            upBound = max_capacity,
                            cat='Continuous') # s_p_j[j,i,k] = start time of chunk k of job j on processor i for predicted jobs

                  e_p_j = pulp.LpVariable.dicts("e_p_j",
                            ((j,p,k) for j in pred_jobs for p in self.hw.processors for k in range(chunk_no_pred)),
                            lowBound = 0,
                            upBound = max_capacity,
                            cat='Continuous') # e_p_j[j,i,k] = end time of chunk k of job j on processor i for predicted jobs

                  b_p = pulp.LpVariable.dicts("b_p",
                            ((p,j1,j2, k1, k2) for p in self.hw.processors for j1 in pred_jobs for j2 in pred_jobs for k1 in range(chunk_no_pred) for k2 in range(chunk_no_pred)),
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger) # binary variable used for taking care of not overlapping chunks of predicted jobs on the same processor
                  

              else: # prediction_intervene == True
                  
                  #### varibale for the case that no predicted task is mapped on resource #######

                  xa_comp_xpred_comp = pulp.LpVariable.dicts("xa_comp_xpred_comp", ((p) for p in gpu_processors), lowBound = 0, upBound = 1, cat=pulp.LpInteger) # xa_comp_xpred_comp[i] = xa_comp[i] * (1-or_x_preds[i])
                  xa_xpred_comp = pulp.LpVariable.dicts("xa_xpred_comp", ((p) for p in gpu_processors), lowBound = 0, upBound = 1, cat=pulp.LpInteger) # xa_xpred_comp[i] = xa[i] * (1-or_x_preds[i])
                  xnp_com_xpred_comp = pulp.LpVariable.dicts("xnp_com_xpred_comp", ((p) for p in gpu_processors), lowBound = 0, upBound = 1, cat=pulp.LpInteger) # xnp_com_xpred_comp[i] = xnp_com[i] * (1-x[nonpreemptable_jobs_on_proc_dict[i],i]), only for gpu processors and nonpreemptable job running on it

                  #### varibale for the case that a predicted task is mapped on resource #######
                  chunk_no = len(prediction_df) + 1
                  x_xpred =  pulp.LpVariable.dicts("x_xpred",
                        ((j,p) for j in jobs for p in self.hw.processors),
                        lowBound = 0,
                        upBound = 1,
                        cat = pulp.LpInteger) # x_xpred[j,i] = x[j,i] * or_x_preds[i]
                  
                  s_j = pulp.LpVariable.dicts("s_j",
                        ((j,p,k) for j in jobs for p in self.hw.processors for k in range(chunk_no)),
                        lowBound = 0,
                        upBound = max_capacity,
                        cat='Continuous') # s[j,i,k] = start time of chunk k of job j on processor i
               
                  e_j = pulp.LpVariable.dicts("e_j",
                        ((j,p,k) for j in jobs for p in self.hw.processors for k in range(chunk_no)),
                        lowBound = 0,
                        upBound = max_capacity,
                        cat='Continuous') # e[j,i,k] = end time of chunk k of job j on processor i
                  
                  b = pulp.LpVariable.dicts("b",
                                     ((p, j1, j2, k1, k2) for p in self.hw.processors for j1 in jobs for j2 in jobs for k1 in range(chunk_no) for k2 in range(chunk_no)),
                                     lowBound = 0,
                                     upBound = 1,
                                     cat = pulp.LpInteger) # binary variable used for taking care of not overlapping chunks of predicted jobs on the same processor
                  
                  
                  q = pulp.LpVariable.dicts("q",
                        ((j,p) for j in jobs for p in gpu_processors),
                        lowBound = 0,
                        upBound = 1,
                        cat = pulp.LpInteger) #   q[j,i] = 0 if job j is not reseted on i, otherwise 1. only for gpu processors
                  
                  x_q = pulp.LpVariable.dicts("x_q",
                        ((j,p) for j in jobs for p in gpu_processors),
                        cat = 'Binary') # x_q[j,i] = x[j,i] * q[j,i]

                  x_q_comp = pulp.LpVariable.dicts("x_q_comp",
                        ((j,p) for j in jobs for p in gpu_processors),
                        lowBound = 0,
                        upBound = 1,
                        cat = pulp.LpInteger) # x_q[j,i] = x[j,i] * (1-q[j,i])
                  
                  x_xpred_q = pulp.LpVariable.dicts("x_xpred_q",
                        ((j,p) for j in jobs for p in gpu_processors),
                        cat = 'Binary') # x_xpred_q[j,i] = x_xpred[j,i] * q[j,i]

          # Step 3: Define the objective function
          if len(prediction_df) > 1 and not prediction_intervene:
            model += (
                    pulp.lpSum([x[(row['key'], p)] * self.rp.energy_consumption(row,p) for j,row in preemptable_jobs.iterrows() for p in self.hw.processors]) #other jobs (preemptable)
                    + pulp.lpSum(xa[row['proc']] *  self.rp.energy_consumption(row, row['proc'], 'reset') for j,row in self.current_mode_nonpreemptable_jobs_df.iterrows()) #restarted on the same processor for nonpreemptable jobs
                    + pulp.lpSum(xa_comp[row['proc']] *  self.rp.energy_consumption(row, row['proc']) for j,row in self.current_mode_nonpreemptable_jobs_df.iterrows()) #continued on the same processor for nonpreemptable jobs
                    + pulp.lpSum(x[(row['key'], p)] * self.rp.energy_consumption(row, p, 'check_migrated') for j,row in self.current_mode_nonpreemptable_jobs_df.iterrows() for p in self.hw.processors) # mapped on other processors for nonpreemptable jobs
                    + pulp.lpSum(y[(p)] for p in self.hw.processors) * self.hw.static_power_consumption)
                               
                    
          else:
              if prediction_intervene:
                model += (
                        pulp.lpSum([x[(row['key'], p)] * self.rp.energy_consumption(row, p) for j,row in preemptable_jobs.iterrows() for p in self.hw.processors]) #other jobs (preemptable)
                        + pulp.lpSum(xa[row['proc']] *  self.rp.energy_consumption(row, row['proc'], 'reset') for j,row in self.current_mode_nonpreemptable_jobs_df.iterrows()) #restarted on the same processor for nonpreemptable jobs
                        + pulp.lpSum(xa_comp[row['proc']] *  self.rp.energy_consumption(row, row['proc']) for j,row in self.current_mode_nonpreemptable_jobs_df.iterrows()) #continued on the same processor for nonpreemptable jobs
                        + pulp.lpSum(x[(row['key'], p)] * self.rp.energy_consumption(row, p, 'check_migrated') for j,row in self.current_mode_nonpreemptable_jobs_df.iterrows() for p in self.hw.processors) # mapped on other processors for nonpreemptable jobs
                        + pulp.lpSum(y[(p)] for p in self.hw.processors) * self.hw.static_power_consumption)
                                 
                            
                        
                
              else:
                 model += (
                        pulp.lpSum([x[(row['key'], p)] * self.rp.energy_consumption(row, p) for j,row in preemptable_jobs.iterrows() for p in self.hw.processors]) #other jobs (preemptable)
                        + pulp.lpSum(xa[row['proc']] *  self.rp.energy_consumption(row, row['proc'], 'reset') for j,row in self.current_mode_nonpreemptable_jobs_df.iterrows()) #restarted on the same processor for nonpreemptable jobs
                        + pulp.lpSum(xa_comp[row['proc']] *  self.rp.energy_consumption(row, row['proc']) for j,row in self.current_mode_nonpreemptable_jobs_df.iterrows()) #continued on the same processor for nonpreemptable jobs
                        + pulp.lpSum(x[(row['key'], p)] * self.rp.energy_consumption(row, p, 'check_migrated') for j,row in self.current_mode_nonpreemptable_jobs_df.iterrows() for p in self.hw.processors) # mapped on other processors for nonpreemptable jobs
                        + pulp.lpSum(y[(p)] for p in self.hw.processors) * self.hw.static_power_consumption
                        )


          # Step 4: Define the constraints

          # constraints (1): each job is mapped to one processor
          for j in jobs:
               model += pulp.lpSum([x[(j, p)] for p in self.hw.processors]) == 1


          # constraints (2): or_x_preds[p] = 1 if at least a predicted job is mapped on p (or linearization)
          if len(prediction_df):
                for p in self.hw.processors:
                    model += pulp.lpSum([x[(j, p)] for j in pred_jobs]) >= or_x_preds[p]
                    for j in pred_jobs:
                        model += or_x_preds[p] >= x[(j, p)]
                     

          # constraints (3): assign values to xa and xa_comp
          for p in gpu_processors:
               #a[p] = 1 if the processor is gpu and the nonpreemptive job is restarted again on this same GPU               
               if nonpreemptable_jobs_on_proc_dict[p] != -1:
                   model += xa[p] <= x[(nonpreemptable_jobs_on_proc_dict[p],p)] 
                   model += xa[p] <= a[p] 
                   model += xa[p] >= a[p] + x[(nonpreemptable_jobs_on_proc_dict[p],p)] - 1 
                 
                   
                   model += xa_comp[p] <= x[(nonpreemptable_jobs_on_proc_dict[p],p)] 
                   model += xa_comp[p] <= (1-a[p])
                   model += xa_comp[p] >= (1-a[p]) + x[(nonpreemptable_jobs_on_proc_dict[p],p)] - 1 
                   

          # constraints (4): schedulability constrains on each processor!
          for p in self.hw.processors: #the main loop for the rest of constraints
              rowp = self.current_mode_nonpreemptable_jobs_df[self.current_mode_nonpreemptable_jobs_df['key'] == nonpreemptable_jobs_on_proc_dict[p]].squeeze() #nonpreemptable job on p if exists

             
              if not prediction_intervene: # prediction does not intervene
                  
                  # schedulability constraints for the nonpredicted jobs
                  for i, row in current_mode_df.iterrows(): 

                    deadline_i = row['deadline']
                    left_time = max(deadline_i - self.current_time, 0)

                    if axf.strip_digits(p) == "cpu" or nonpreemptable_jobs_on_proc_dict[p] == -1:
                        model +=  pulp.lpSum([(x[(rowj['key'], p)] * self.rp.execution_time(rowj, p)) for j,rowj in current_mode_df[:(i+1)].iterrows()]) <= left_time 
                        
                    else: # for GPU processors with nonpreemptable job running on it
                        tmp_df = current_mode_df[0:i+1]
                        jobs_minus_nonpreeptables_proc_i_df = tmp_df[~tmp_df['key'].isin([nonpreemptable_jobs_on_proc_dict[p]])].reset_index(drop=True)
                        model += pulp.lpSum([(x[rowj['key'], p] * self.rp.execution_time(rowj, p)) for j,rowj in jobs_minus_nonpreeptables_proc_i_df.iterrows()]) + self.rp.execution_time(rowp, p) * x[row['key'], p] <= (-left_time * (((-1 * M) * (1 - xa_comp[p])) - 1) + M * (1 - xa_comp[p]))
                        model += pulp.lpSum([(x[rowj['key'], p] * self.rp.execution_time(rowj, p, 'reset')) for j,rowj in current_mode_df[0:i+1].iterrows()]) <= -left_time * (((-1 * M) * (1 - xa[p])) - 1) 
                        model += pulp.lpSum([(x[rowj['key'], p] * self.rp.execution_time(rowj, p)) for j,rowj in current_mode_df[0:i+1].iterrows()]) <= -left_time * (x[nonpreemptable_jobs_on_proc_dict[p],p] - 1) + M * x[nonpreemptable_jobs_on_proc_dict[p],p]
                    


                  # schedulability constraints for the predicted jobs
                  if len(prediction_df):
                      
                      if axf.strip_digits(p) == "cpu" or nonpreemptable_jobs_on_proc_dict[p] == -1:
                        model += f[p] == pulp.lpSum([(x[(row['key'], p)] * self.rp.execution_time(row, p)) for i, row in current_mode_df.iterrows()])    #f[p] is the end of execution times of the nonpredicted job on p                   
                           
                      else: # for GPU processors with nonpreemptable job running on it
                        preemptable_jobs_on_proc_df = current_mode_df[~current_mode_df['key'].isin([nonpreemptable_jobs_on_proc_dict[p]])].reset_index(drop=True) 
                        model += f[p] == self.rp.execution_time(rowp, p) * xa_comp[p] + self.rp.execution_time(rowp, p, 'reset') * xa[p] + pulp.lpSum([(x[(row['key'], p)] * self.rp.execution_time(row, p)) for j, row in preemptable_jobs_on_proc_df.iterrows()])  #nonpreemptable job on p mighe be restarted again on p or not restarted at all , f[p] is the end of execution times of the nonpredicted job on p                                     

                      # predicted tasks might be broken into multiple parts
                      num_chunck = defaultdict(lambda: 1)
                      if axf.strip_digits(p) == "cpu" and len(prediction_df)>1:

                        prediction_sorted_by_arrival_df = prediction_df.sort_values(by='arrival_time').reset_index(drop = True) #revise 
                        for i, row in prediction_sorted_by_arrival_df.iterrows():
                            key = row['key']
                            for j, rowj in prediction_sorted_by_arrival_df[i+1:].iterrows():
                                if row['deadline'] > rowj['deadline'] and row['arrival_time'] <= rowj['arrival_time']:
                                    num_chunck[key] += 1

                      for i, row in prediction_df.iterrows():

                            # multiplication linearization
                            key = row['key']
                            lower_band = 0
                            upper_band = max_capacity
                            model += f[p] >= lower_band
                            model += f[p] <= upper_band
                            model += fx[(key,p)] >= lower_band * x[(key,p)]
                            model += fx[(key,p)] <= upper_band * x[(key,p)]
                            model += fx[(key,p)] <= f[p] - lower_band * (1 - x[(key,p)])
                            model += fx[(key,p)] >= f[p] - upper_band * (1 - x[(key,p)])
            
                            # start time constraints
                            model += s_p_j[(key, p, 0)] >= (row['arrival_time'] - self.current_time) * x[(key,p)] # start time of the first chunk of the predicted job should be equal or greater than the arrival time of the job
                            model += s_p_j[(key, p, 0)] >= fx[(key,p)] # start time of the first chunk of the predicted job should be equal or greater than the end of execution time of the nonpredicted job
                            
                            
                            for k in range(chunk_no_pred):
                                if k < num_chunck[key]:
                                    model += e_p_j[(key, p, k)] <= M * x[(key,p)]
                                    model += s_p_j[(key, p, k)] <= M * x[(key,p)]
                                    model += e_p_j[(key, p, k)] - s_p_j[(key, p, k)] >= M * (x[(key,p)] - 1) # end time of the chunk should be greater or equal than the start time of the chunk
                                    
                                    if k < num_chunck[key] - 1:
                                        model += s_p_j[(key, p, k+1)] - s_p_j[(key, p, k)] >= M * (x[(key,p)] - 1) # start time of the next chunk should be equal or greater than the end time of the current chunk
                                        model += s_p_j[(key, p, k+1)] - e_p_j[(key, p, k)] >= M * (x[(key,p)] - 1) # start time of the next chunk should be equal or greater than the end time of the current chunk
                                        

                                    if k == num_chunck[key] - 1:
                                        model += e_p_j[(key, p, k)] <= (row['deadline'] - self.current_time) * x[(key,p)] # end time of the last chunk should be equal or less than the deadline of the job
                                        
                                else:
                                    
                                    model += e_p_j[(key, p, k)] == 0
                                    model += s_p_j[(key, p, k)] == 0
                                    

                            model += pulp.lpSum(e_p_j[(key, p, k)] - s_p_j[(key, p, k)] for k in range(chunk_no_pred)) == self.rp.execution_time(row, p) * x[(key,p)] # the sum of the execution times of the chunks should be equal to the execution time of the job
                            

                            # the chunks should not overlap on the same processor
                            for j, rowj in prediction_df.iterrows():
                                key1 = rowj['key']
                                if key1 == key:
                                    continue
                                for k in range(chunk_no_pred): 
                                    for k1 in range(chunk_no_pred):
                                        model += e_p_j[(key, p, k)] - s_p_j[(key1, p, k1)] <=  M * b_p[(p, key, key1, k, k1)]
                                        model += e_p_j[(key1, p, k1)] - s_p_j[(key, p, k)] <=  M * (1 - b_p[(p, key, key1, k, k1)])
                                          

              else: # prediction intervenes
                
                ######## in the case that no predicted job is mapped on p ######
                ####################################################################
                for i, row in self.rp.all_jobs_df.iterrows():
                       

                       deadline_i = row['deadline']
                       left_time = max(deadline_i - self.current_time, 0)   
                    
                       if axf.strip_digits(p) == "cpu" or nonpreemptable_jobs_on_proc_dict[p] == -1:
                           model += pulp.lpSum([(x[(rowj['key'], p)] * self.rp.execution_time(rowj, p)) for j,rowj in self.rp.all_jobs_df[:i+1].iterrows()]) <=   -left_time * ((-1 * M) * or_x_preds[p] - 1) 
                           

                       else: # for GPU processors with nonpreemptable job running on it
                           
                           # multiplication linearization
                           model += xa_comp_xpred_comp[p] <= (1-or_x_preds[p])
                           model += xa_comp_xpred_comp[p] <= xa_comp[p]
                           model += xa_comp_xpred_comp[p] >= xa_comp[p] + (1-or_x_preds[p]) - 1 
                         
                           
                           # multiplication linearization
                           model += xa_xpred_comp[p] <= (1-or_x_preds[p])
                           model += xa_xpred_comp[p] <= xa[p]
                           model += xa_xpred_comp[p] >= xa[p] + (1-or_x_preds[p]) - 1 
                         
                           
                           # multiplication linearization
                           model += xnp_com_xpred_comp[p] <= (1-or_x_preds[p])
                           model += xnp_com_xpred_comp[p] <= (1-x[(nonpreemptable_jobs_on_proc_dict[p],p)])
                           model += xnp_com_xpred_comp[p] >= (1-or_x_preds[p] ) + (1-x[(nonpreemptable_jobs_on_proc_dict[p],p)]) - 1 
                          
                           
                           tmp_df = self.rp.all_jobs_df[0:i+1]
                           jobs_minus_nonpreeptables_proc_i_df = tmp_df[~tmp_df['key'].isin([nonpreemptable_jobs_on_proc_dict[p]])].reset_index(drop=True)
                           
                           model += pulp.lpSum([(x[rowj['key'], p] * self.rp.execution_time(rowj, p)) for j,rowj in jobs_minus_nonpreeptables_proc_i_df.iterrows()]) + self.rp.execution_time(rowp, p) * x[row['key'], p] <= -left_time * (((-1 * M) * (1 - xa_comp_xpred_comp[p])) - 1) + M * (1 - xa_comp_xpred_comp[p]) 
                           model += pulp.lpSum([(x[rowj['key'], p] * self.rp.execution_time(rowj, p, 'reset')) for j,rowj in self.rp.all_jobs_df[0:i+1].iterrows()]) <= -left_time * (((-1 * M) * (1 - xa_xpred_comp[p])) - 1) 
                           model += pulp.lpSum([(x[rowj['key'], p] * self.rp.execution_time(rowj, p)) for j,rowj in self.rp.all_jobs_df[0:i+1].iterrows()]) <= left_time * (xnp_com_xpred_comp[p]) + M * (1-xnp_com_xpred_comp[p])
                         


                ######## in the case that the predicted job is mapped on p #########
                ###################################################################
                

                if axf.strip_digits(p) == "cpu":
                    num_chunck = defaultdict(lambda: chunk_no)
                else: #gpu is not nonpreemptive
                    num_chunck = defaultdict(lambda: 1)

              
                for i, row in self.rp.all_jobs_df.iterrows(): 

                        key = row['key']

                        ### multiplication linearization
                        model += x_xpred[(key,p)] <= or_x_preds[p]
                        model += x_xpred[(key,p)] <= x[(key,p)]
                        model += x_xpred[(key,p)] >= x[(key,p)] + or_x_preds[p] - 1 
            

                        if key in pred_jobs:    
                            model += s_j[(key, p, 0)] >= (row['arrival_time'] - self.current_time) * x_xpred[(key,p)]  # start time of the first chunk should be equal or greater than the arrival time of the job
                       
                        for k in range(chunk_no):
                            if k < num_chunck[key]:
                                model += e_j[(key, p, k)] <= M * x_xpred[(key,p)] 
                                model += s_j[(key, p, k)] <= M * x_xpred[(key,p)] 
                                model += e_j[(key, p, k)] - s_j[(key, p, k)] >= M * (x_xpred[(key,p)] - 1) # end time of the chunk should be equal or greater than the start time of the chunk
                                
                                if k < num_chunck[key] - 1:
                                    model += s_j[(key, p, k+1)] - s_j[(key, p, k)] >= M * (x_xpred[(key,p)] - 1) # start time of the next chunk should be equal or greater than the start time of the current chunk
                                    model += s_j[(key, p, k+1)] - e_j[(key, p, k)] >= M * (x_xpred[(key,p)] - 1) # start time of the next chunk should be equal or greater than the end time of the current chunk
                                    

                                if k == num_chunck[key] - 1:
                                    model += e_j[(key, p, k)] <= (row['deadline'] - self.current_time) * x_xpred[(key,p)] # end time of the last chunk should be equal or less than the deadline of the job
                                    
                            else:
                                
                                model += e_j[(key, p, k)] == 0
                                model += s_j[(key, p, k)] == 0
                               

                        if axf.strip_digits(p) == "cpu":    
                            model += pulp.lpSum(e_j[(key, p, k)] - s_j[(key, p, k)] for k in range(chunk_no)) == self.rp.execution_time(row, p) * x_xpred[(key,p)] # execution time of the job on p should be equal to the sum of the execution time of the chunks on p
                            
                        else: # for GPUs

                            ## if s_j[(key, p, 0)] = 0 --> q[(key,p)] = 0 (not resetarted),and if s_j[(key, p, 0)] > 0 --> q[(key,p)] = 1 (resetarted)
                            model += q[(key,p)] <= (s_j[(key, p, 0)] * M)
                            model += q[(key,p)] >= (s_j[(key, p, 0)] * (1/max_capacity))

                            # multiplication linearization
                            model += x_xpred_q[(key,p)] <= x_xpred[(key,p)] 
                            model += x_xpred_q[(key,p)] <= q[(key,p)]
                            model += x_xpred_q[(key,p)] >= x_xpred[(key,p)] + q[(key,p)] - 1  
                      

                            # multiplication linearization
                            model += x_q[(key,p)] <= x[(key,p)] 
                            model += x_q[(key,p)] <= q[(key,p)]
                            model += x_q[(key,p)] >= x[(key,p)] + q[(key,p)] - 1 
                            

                            # multiplication linearization
                            model += x_q_comp[(key,p)] <= x[(key,p)]
                            model += x_q_comp[(key,p)] <= (1-q[(key,p)])
                            model += x_q_comp[(key,p)] >= x[(key,p)] + (1-q[(key,p)]) - 1 
                           
                            if key == nonpreemptable_jobs_on_proc_dict[p]:
                                model += xa[p] <=  (x_q[(key,p)] + M * (1-or_x_preds[p])) 
                                model += xa_comp[p] <= (x_q_comp[(key,p)] + M * (1-or_x_preds[p])) 
                              

                            ## if q[(key,p)] = 1 --> reset
                            
                            model += pulp.lpSum(e_j[(key, p, k)] - s_j[(key, p, k)] for k in range(chunk_no)) <= (self.rp.execution_time(row, p) * x_xpred[(key,p)] + self.rp.execution_time(row, p) * M * x_xpred_q[(key,p)])
                            model += pulp.lpSum(e_j[(key, p, k)] - s_j[(key, p, k)] for k in range(chunk_no)) <= (self.rp.execution_time(row, p, 'reset') * x_xpred[(key,p)] * (M+1) - self.rp.execution_time(row, p, 'reset') * M * x_xpred_q[(key,p)]) 
                            model += pulp.lpSum(e_j[(key, p, k)] - s_j[(key, p, k)] for k in range(chunk_no)) >= (self.rp.execution_time(row, p) * x_xpred[(key,p)] - self.rp.execution_time(row, p) * x_xpred_q[(key,p)])
                            model += pulp.lpSum(e_j[(key, p, k)] - s_j[(key, p, k)] for k in range(chunk_no)) >= self.rp.execution_time(row, p, 'reset') * x_xpred_q[(key,p)] 

                        ## chunks on the same resource should not overlap
                        for j,rowj in self.rp.all_jobs_df.iterrows(): 
                            key1 = rowj['key']
                            if key1 == key:
                                continue
                            for k in range(chunk_no): 
                                for k1 in range(chunk_no):
                                    model += e_j[(key, p, k)] - s_j[(key1, p, k1)] <=  M * b[(p,key, key1, k, k1)]
                                    model += e_j[(key1, p, k1)] - s_j[(key, p, k)] <=  M * (1 - b[(p,key, key1, k, k1)])
                

                  

          # end of constraints (4) (schedulability constrains)



          # Step 5: Solve the problem and retrieve the results

          if gurobi_solver:
            # gurobi solver
            solver = pulp.GUROBI_CMD(options=[("timeLimit", 100)],msg = False)
            model.solve(solver)
            status = pulp.LpStatus[model.status]

          else:
            # CBC solver
            solver = pulp.PULP_CBC_CMD()
            model.solve(solver)
            status = pulp.LpStatus[model.status]
          


          # Step 6: check status of the problem
          if status != "Optimal":
              if len(prediction_df): # drop prediction 
                  prediction_df = pd.DataFrame(columns = self.rp.all_jobs_df.columns)

              else:
                  break
          
       # end of while loop (while status != "Optimal")


       # if write_schedule is True, write the schedule to a file
       if write_schedule:
           if status == "Optimal":
               
               #self.rp.all_jobs_df['prev_proc'] = self.rp.all_jobs_df['proc']
               # first write mapping
               for var in x:
                    var_value = x[var].varValue
                    
                    if var_value and var_value > 0.9:
                        self.rp.all_jobs_df.loc[self.rp.all_jobs_df['key'] == var[0], 'proc'] = var[1]                        
                            
                # then write schedule
               if prediction_intervene:
                   self.write_schedule(nonpreemptable_jobs_on_proc_dict, priority_jobs_df, s_j, e_j, x , a, xa_comp, prediction_intervene)
               else:
                   if len(prediction_df):
                       self.write_schedule(nonpreemptable_jobs_on_proc_dict, priority_jobs_df, s_p_j, e_p_j, x , a, xa_comp, True) 
                   else:
                       self.write_schedule(nonpreemptable_jobs_on_proc_dict, priority_jobs_df, [], [], x , a, xa_comp, prediction_intervene) 
                       
                       
                   
       
       
       return "Optimal" if status == "Optimal" else "Infeasible"
    


    def write_schedule(self, nonpreemptable_dic, priority_jobs_df, s_j, e_j , x, a, xa_comp, prediction_intervene):
     
        """
        This function writes the schedule to a file if the optimization problem is solved. It takes in the following arguments:


        Parameters:
        -----------
            - nonpreemptable_dic: a dictionary of non-preemptable jobs 
            - priority_jobs_df: a DataFrame containing the priority jobs
            - s_j: a dictionary of decision variables representing the start time of each chunk of each job on each processor
            - e_j: a dictionary of decision variables representing the end time of each chunk of each job on each processor
            - x: a dictionary of binary decision variables representing whether a job is mapped to a processor
            - a: a dictionary of binary decision variables representing whether a processor is a GPU and a non-preemptive job is restarted again on the same GPU
            - xa_comp: a dictionary of binary decision variables, xa_comp[(j, i)] = 1 if job j is mapped to processor i and a[i] = 0
            - prediction_intervene: a boolean variable indicating whether prediction intervene or not

        Returns:
            - None
        """



        #### initialization #####

        nonpreemptable_jobs_keys = list(set(self.current_mode_nonpreemptable_jobs_df['key']))
        pred_job = self.rp.all_jobs_df[self.rp.all_jobs_df['is_pred']==1]

        # Create a dataframe to store the schedule 
        schedule = pd.DataFrame(columns=self.rp.all_jobs_df.columns) 

        # If there is no job, then write an empty schedule
        if not len(self.rp.all_jobs_df):
            schedule.to_csv('schedule.csv')
            return 0

        
        execution_time_sum = defaultdict(lambda:0) # a variable to take care of scheduling time for jobs on the processors

        Jobs_written_to_schedule_list = [] # a list to keep track of jobs that are written to the schedule

        #### end of initialization #####


        # Step 1: write the schedule for nonpreemptable jobs
        for i, row in self.current_mode_nonpreemptable_jobs_df.iterrows():
            key = row['key'] 
            proc = self.rp.all_jobs_df[self.rp.all_jobs_df['key'] == key]['proc'].values[0] 
            try: #it might be migrated to other processor and both xa_comp and xa be invalid!!
                if xa_comp[proc].varValue and xa_comp[proc].varValue > 0.9:
                    Jobs_written_to_schedule_list.append(key)
                    row['start'] = execution_time_sum[proc]
                    execution_time_sum[proc] += self.rp.execution_time(row ,proc,'',True)
                    row['end'] =  execution_time_sum[proc]
                    schedule = pd.concat([schedule, row.to_frame().transpose()], ignore_index = True)
            except:
                pass
            

        # Step2: write the schedule for the ones that might have multiple chunks
        if prediction_intervene:
           for var1, var2 in zip(s_j,e_j):
               var_value_s = s_j[var1].varValue
               var_value_e = e_j[var2].varValue
               if var1[0] != var2[0] or var1[2] != var2[2]:
                   raise Exception('var1[0] != var2[0] or var1[2] != var2[2] in write_schedule fuction')
               
               if (var_value_s + var_value_e > 0) and (int(var_value_s*10000) != int(var_value_e*10000)):
                    
                    if var1[0] in Jobs_written_to_schedule_list and var1[0] in nonpreemptable_jobs_keys:
                        continue 

                    Jobs_written_to_schedule_list.append(var1[0])
                    tmp = self.rp.all_jobs_df[self.rp.all_jobs_df['key'] == var1[0]].copy()
                
                    tmp['proc'] = var1[1] 
                    tmp['start'] = var_value_s 
                    tmp['end'] = var_value_e 
                    tmp['chunk'] = var1[2] 
                    
                    schedule = pd.concat([schedule,tmp], ignore_index = True)   
                    
        
        # Step 3: write the schedule for priority_jobs_df
        for i, row_tmp in priority_jobs_df.iterrows():
            
            key = row_tmp['key'] 
            row = self.rp.all_jobs_df.loc[self.rp.all_jobs_df['key'] == key].squeeze()
            if key in Jobs_written_to_schedule_list:
                continue
            Jobs_written_to_schedule_list.append(key)
            proc = self.rp.all_jobs_df[self.rp.all_jobs_df['key'] == key]['proc'].values[0] 
            if not row['is_pred']:
                row['start'] = execution_time_sum[proc]
                execution_time_sum[proc] += self.rp.execution_time(row ,proc, 'reset',True) 
                row['end'] = execution_time_sum[proc]      
                
            else:
                if row['arrival_time'] - self.current_time <= execution_time_sum[proc]: 
                    row['start'] = execution_time_sum[proc]
                    execution_time_sum[proc] += self.rp.execution_time(row ,proc,'',True)
                    row['end'] = execution_time_sum[proc] 
                    
                    
                else:
                    row['start'] = row['arrival_time'] - self.current_time  
                    execution_time_sum[proc] = row['arrival_time'] - self.current_time + self.rp.execution_time(row ,proc,'',True) 
                    end_time = row['end'] = execution_time_sum[proc] 
                    
            schedule = pd.concat([schedule,row.to_frame().transpose()], ignore_index = True)
               
        
        #Step 4: wirte the schedule for the rest of the jobs     
        rest_jobs_df = pd.DataFrame(columns = self.rp.all_jobs_df.columns)          
        Jobs_written_to_schedule_list = list(set(Jobs_written_to_schedule_list))
        for var in x:
           var_value = x[var].varValue
           if var_value and var_value > 0.9:
               if var[0] not in Jobs_written_to_schedule_list:
                   rest_jobs_df = pd.concat([rest_jobs_df,self.rp.all_jobs_df[self.rp.all_jobs_df['key'] == var[0]].copy()], ignore_index = True)
        
                   
        rest_jobs_df.sort_values(by='deadline', inplace=True, ignore_index=True)
        for i,row in rest_jobs_df.iterrows():
           
            key = row['key'] 
            proc = self.rp.all_jobs_df[self.rp.all_jobs_df['key'] == key]['proc'].values[0]
            if not row['is_pred']: 
                row['start'] = execution_time_sum[proc] 
                execution_time_sum[proc] += self.rp.execution_time(row ,proc, 'reset')
                row['end'] = execution_time_sum[proc] 
                
            else:
                if row['arrival_time'] - self.current_time <= execution_time_sum[proc]: 
                    row['start'] = execution_time_sum[proc] 
                    execution_time_sum[proc] += self.rp.execution_time(row ,proc)
                    row['end'] = execution_time_sum[proc] 
                    
                else:
                    row['start'] = row['arrival_time'] - self.current_time
                    execution_time_sum[proc] = row['arrival_time'] - self.current_time + self.rp.execution_time(row ,proc)
                    row['end'] = execution_time_sum[proc] 
           
            
            schedule = pd.concat([schedule,row.to_frame().transpose()], ignore_index = True) 
            

        # Final step: adjsut the schedule and write it to csv file
        cols = [c for c in schedule.columns if c.lower()[:7] != 'unnamed']
        schedule = schedule[cols]
        schedule = axf.delete_gpu_duplication(schedule)
        schedule = axf.adjust_schedule(schedule,self.current_time)
        cols = [c for c in ['key','arrival_time','deadline','job_name','proc','start','end']]
        schedule = schedule[cols]
        schedule['start'] = schedule['start'] + self.current_time
        schedule['end'] = schedule['end'] + self.current_time
        schedule.sort_values(by = 'start').reset_index(drop = True).to_csv('schedule.csv')
          
       
     
       
      