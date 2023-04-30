
from convert_redis import conver_performance_to_redis,migration_cost_to_redis
from pandas.io.stata import excessive_string_length_error
import redis 
import pandas as pd
from pathlib import Path
from os.path import join as path_join
import os
import sys, getopt
import milp as milp
import heuristic as heu




def check_inputs(current_state_file: str, performance_file: str,solver: str, current_time: float, gurobi_solver: bool, write_schedule_log: bool):
    """
    Checks if the inputs are valid.

    Args:

        current_state_file: Path to the current state file (csv).
        performance_file: Path to the performance file (csv).
        solver: Solver type (either "HEU" or "MILP").
        current_time: Current time value to check for in the "arrival_time" column of the current state file.

    Raises:
        FileNotFoundError: If the current state file or performance file does not exist.
        ValueError: If the solver type is not "HEU" or "MILP", or if the current time is not in the arrival_time column of the current state file.
    """


    assert isinstance(current_state_file, str), "current_state_file should be a string"
    assert isinstance(performance_file, str), "performance_file should be a string"
    assert isinstance(solver, str), "solver should be a string"
    assert isinstance(current_time, float), "current_time should be an integer"

    # Check if current state file exists.
    if not os.path.exists(current_state_file):
        raise FileNotFoundError(f"Current state file {current_state_file} does not exist.")
    
    # Check if performance file exists.
    if not os.path.exists(performance_file):
        raise FileNotFoundError(f"Performance file {performance_file} does not exist.")
    
    # Check if solver is valid.
    if solver not in ["HEU","MILP"]:
        raise ValueError(f"Invalid solver type: {solver}. Solver type should be 'HEU' or 'MILP'.")
    
    # Check if gurobi_solver is valid.
    if gurobi_solver not in [False,True]:
        raise ValueError(f"Invalid bool value for have/not have grubi solver: {gurobi_solver}.  should be 'False' or 'True'.")
    
    # Check if write_schedule_log is valid.
    if write_schedule_log not in [False,True]:
        raise ValueError(f"Invalid bool value for write/not write schedule log: {write_schedule_log}.  should be 'False' or 'True'.")
    

    # Check if current time is valid and available in the "arrival_time" column of the current state file.
    df = pd.read_csv(current_state_file)
    if current_time not in df["arrival_time"].values:
        raise KeyError("time not found in current state file.")
    

def main(argv):
    """
    Main function for running the optimization algorithm.

    Parameters:
    -----------
        argv (list): list of command line arguments

            Command Line Arguments:
            -w (str): current state file (csv)
            -p (str): performance file (csv)
            -s (str): solver (["HEU","MILP"])
            -t (float): current time
            -g (bool): have/not have grubi soliver (["False","True"]) (default: False)
            -l (bool): write/not write schedule log (["False","True"]) (default: False)

    Returns:
    ---------
        srt
            'Optimal' if a feasible solution is found, otherwise returns 'Infeasible'
    """

    # Parse command line arguments
    try:
        opts, args = getopt.getopt(argv, "w:p:s:t:g:l:")
    except getopt.GetoptError as e:
        print("Error: Invalid command line argument.")
        return 1
   
    # Initialize variables
    current_state_file = None
    performance_file = None
    solver = None
    current_time = None
    gurobi_solver = False
    write_schedule_log = False

    for opt, arg in opts:
        if opt == '-w': # current state file
            current_state_file = str(arg)
        if opt == '-p': # performance file
            performance_file = str(arg)
        if opt == '-s': # solver
            solver = str(arg)
        if opt == '-t': # current time
            current_time = float(arg)
        if opt == '-g':
            gurobi_solver = bool(arg)
        if opt == '-l':
            write_schedule_log = bool(arg)

    # Check that all required inputs have been provided
    if None in [current_state_file, performance_file, solver, current_time]:
        print("Missing required input.")
        return 1

    
    # Check that inputs are valid
    check_inputs(current_state_file,performance_file,solver, current_time, gurobi_solver, write_schedule_log)
    
    # Convert performance file to redis
    conver_performance_to_redis(performance_file)

    # Read in performance file and create list of processors
    performace_df = pd.read_csv(performance_file)
    processors_list = list(performace_df.proc.unique()) 

    # Store migration cost data in redis
    migration_cost_to_redis(processors_list)

    # Count the number of CPUs and GPUs
    n_cpu = 0
    n_gpu = 0
    for elem in processors_list:
        if elem.startswith("cpu"):
            n_cpu += 1
        elif elem.startswith("gpu"):
            n_gpu +=1

    # Run the optimization algorithm        
    if solver == "MILP":
        MILP_solver = milp.MILP(current_state_file, performance_file, current_time, n_cpu, n_gpu)
        status = MILP_solver.solve(gurobi_solver = gurobi_solver, write_schedule=write_schedule_log)
    else:
        heu_solver = heu.Heuristic(current_state_file, performance_file, current_time, n_cpu,n_gpu)
        status = heu_solver.solve(write_schedule=write_schedule_log)
        
    print("Optimization status of {} solver: {}".format(solver,status))
    return status
    
    

if __name__ == "__main__":
    main(sys.argv[1:])
