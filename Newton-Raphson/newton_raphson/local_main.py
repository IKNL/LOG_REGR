import server_code
import numpy as np
import json

with open('local_config.json') as config:
    simulated_data = json.load(config)

server_code.calculate_coefficients(beta=np.asmatrix(np.zeros(simulated_data["nodes"]["number_of_parameters"])),
                                   result_file=r"C:\project\simulations\datasets_simulations\datasets_simulations\simulations\NR_results.json")