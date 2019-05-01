import server_code
import numpy as np
import json

with open('local_config.json') as config:
    data = json.load(config)

server_code.calculate_coefficients(beta=np.asmatrix(np.zeros(data["nodes"]["number_of_parameters"])))