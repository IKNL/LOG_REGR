from central_server import Central_Node
import pandas as pd
import random
import os
from pathlib import Path

number_of_selections = 100
initial_seed = 10
random.seed(initial_seed)
seeds = random.sample(range(1, 10000), number_of_selections)
full_data = pd.read_csv("data/full_data.csv")
number_of_nodes = 2
main_file_min_range = 100
selection_increase = 100
cwd = os.getcwd()

for selection in range(0, number_of_selections):
    central_node_records_number = main_file_min_range + selection * selection_increase
    central_data = full_data.sample(n=central_node_records_number, random_state=seeds[selection])
    central_data.to_csv(Path(cwd + "/data/central_data.csv"), index=False)
    data_left = full_data.drop(central_data.index)
    first_file = data_left.sample(n=int(len(data_left) / number_of_nodes), random_state=seeds[selection])
    first_file.to_csv(Path(cwd + "/data/first_file.csv"), index=False)
    second_file = data_left.drop(first_file.index)
    second_file.to_csv(Path(cwd + "/data/second_file.csv"), index=False)
    central = Central_Node(data_file="data/full_data.csv", outcome_variable="3_years_death")
    log_file = Path(cwd + "/simulation_logs/simulation{}.txt".format(selection))
    open(log_file, 'w+').close()
    with open(log_file, "a+") as file:
        file.write("Central server contains {} number of recors\n".format(len(central_data)))
        file.write("First node contains {} number of recors\n".format(len(first_file)))
        file.write("Second node contains {} number of recors\n".format(len(second_file)))
    central.calculate_global_coefficients(log_file)
