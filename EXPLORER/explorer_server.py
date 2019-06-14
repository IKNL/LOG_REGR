import new_job_listener
import threading
import time

clients = 10
central_message = {"iter": 1, "mw": 0, "vw": 5}
results = []

for client in clients:
    thread = threading.Thread(target=new_job_listener.new_job_listener,
                              args=(results, central_message,))
    #TODO implement new_job_listener as a class


while True:
    if(len(results) != clients):
        time.sleep(0.5)

