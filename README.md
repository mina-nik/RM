# Resource Manager

This code provides both the MILP (mixed integer linear programming) and the heuristic algorithm implementations of the resource manager described in our paper . . . 

## Prerequisites
To run the resource manager, you will need:


- [Python 3.x](https://www.python.org/downloads/)
- [Redis](https://redis.io/download)
- [Pulp](https://pypi.org/project/PuLP/)
- [Gurobi](https://www.gurobi.com/documentation/) (optional, used for more complex MILP problems)



## Usage
To run the resource manger, navigate to the project root directory and run the following command:

```python
python runner.py   -p <performance_file>  -w <current_state_file> -t <current_time> -g <use_gurobi_solver> -l <create_schedule_log>
```

Replace <performance_file>, <current_state_file>, <current_time>, <use_gurobi_solver>, and <create_schedule_log> with the appropriate values. Here is an explanation of each command line argument:

- -p: specifies the file that contains the execution times and energy consumption of the tasks on the resources in a CSV file.
- -w: specifies the current state of the workload in a CSV file.
- -t: specifies the current scheduling time (this value should be one of the arrival times in <current_state_file>)
- -g: specifies whether to use the Gurobi solver for the MILP or not. Use 'True' to enable (default is false).
- -l: specifies whether to create a schedule log as an output CSV file. Use 'True' to enable (default is false).

## File Formats

Detailed explanation of file formats:

### <performance_file>: csv
This CSV file contains information about the execution times and energy consumption of each task on the available resources (each line specifies the execution time and energy consumption of a task on a processor):

- 'job_name': a custom name for the job associated with the task.
- 'proc': the processor name.
- 'exe_time': the execution time of 'job_name' on 'proc'. 
- 'energy': the amount of energy consumed by 'job_name' when executed on 'proc'.


As an example see perf.csv.


### '<current_state_file>': csv file:
This CSV file contains information about the tasks to be scheduled. Each line corresponds to the specification of each request/task:

- 'arrival_time': the time at which the task arrives.
- 'deadline': the latest time by which the task must be completed.
- 'job_name': the name of the task (this value should be one of the job names included in <performance_file>).
- 'key': a unique identifier for the task.
- 'proc': the name of the processor (e.g., gpu1) on which the task has been mapped before the current scheduling time (the name of the processor should be one of the processor names in <performance_file> or None (in case it is not mapped yet)).
- 't_mig': the migration overhead in terms of time if the task has been already relocated from another processor to 'proc'.
- 'e_mig': the migration overhead in terms of energy if the task has been already relocated from another processor to 'proc'.
- 'progress': the current progress of execution of task untill current scheduling time, expressed as a fraction between 0 and 1.

As an example see state.csv.




## Usage Example

Here are an examples of how to run the resource manager:
```python
python runner.py  -p perf.csv -w state.csv  -t 300 
```



