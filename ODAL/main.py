from central_server import Central_Node
import numpy as np


central = Central_Node(data_file="data/full_data.csv", outcome_variable="3_years_death")
central.get_global_coefficients()