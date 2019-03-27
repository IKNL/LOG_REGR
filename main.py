""" Main.py

This is the main entry-point when the docker-container is initialized.
It executes the following steps:

1) read the input.txt
    This should contain the name of the method that should be executed.
    Optionally it contains some args and kwargs
2) In case it is a master algorithm the token is read from token.txt
    This is a JWT token that can be used to interact with the server
3) The method is executed
4) The output it written to output.txt

If the docker container is terminated. Output.txt will be send to the
server by the node.
"""
import json
import os
import sys

from server_code import calculate_coefficients, info, warn
from node_code import make_local_iteration

# read input from the mounted inputfile
info("Reading input")
with open("app/input.txt") as fp:
    input_ = json.loads(fp.read())

# determine function from input, summarize is used by default.
# and get the args and kwargs input for this function
method_name = input_.get("method","summary")
method = {
    "node_regression": make_local_iteration,
    "master": calculate_coefficients
}.get(method_name)
if not method:
    warn(f"method name={method_name} not found!\n")
    exit()

args = input_.get("args", [])
kwargs = input_.get("kwargs", {})

# call function
if method_name == "master":
    info("Reading token")
    with open("app/token.txt") as fp:
        token = fp.read().strip()
        info(token)
    output = method(token, *args, **kwargs)
else:
    output = method(*args, **kwargs)

# write output to mounted output file
info("Writing output")
with open("app/output.txt", 'w') as fp:
    fp.write(json.dumps(output))
