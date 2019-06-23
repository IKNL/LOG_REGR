from explorer_node_operations import train_linear, gaussian_division
import pandas as pd
import numpy as np

class explorer_node():
    new_iteration = False
    close = False
    id = -1
    central_message = {}
    XTR = pd.DataFrame()
    YTR = pd.DataFrame()

    def __init__(self, id, XTR, YTR):
        self.id =id
        self.XTR = XTR.copy()
        self.YTR = YTR.copy()
        self.YTR.loc[self.YTR == 0] = -1
        self.YTR = np.array(self.YTR)
        self.YTR = self.YTR.reshape(self.YTR.shape + (1,))



    def run_one_more_iteration(self, central_message):
        self.new_iteration = True
        self.central_message = central_message


    def stop_computations(self):
        self.close = True


    # in the original data is stored inside the sockets, here connection is modified so data is accepted instead of
    def new_job_listener(self, results):
        EP_result = {}
        EP_result["obj_new"] = {}
        EP_result["incoming_mw"] = []
        EP_result["incoming_vw"] = []
        EP_result["outgoing_mw"] = []
        EP_result["outgoing_vw"] = []
        EP_result["obj_new"]["state"] = []
        while not self.close:
            if self.new_iteration:
                EP_result["iter"] = self.central_message["iter"]
                EP_result["obj"] = EP_result["obj_new"]
                EP_result["obj_new"] = train_linear(self.XTR, self.YTR, "logistic",
                                                    self.central_message["mw"], self.central_message["vw"],
                                                    EP_result)
                if EP_result["iter"] == 1:
                    EP_result["outgoing_mw"] = EP_result["obj_new"]["mw"]
                    EP_result["outgoing_vw"] = EP_result["obj_new"]["vw"]
                else:
                    EP_result.incoming_mw, EP_result.incoming_vw = gaussian_division(
                        self.central_message["mw"],
                        self.central_message["vw"],
                        EP_result["outgoing_mw"],
                        EP_result["outgoing_vw"])
                    EP_result.outgoing_mw, EP_result.outgoing_vw = gaussian_division(
                        EP_result["obj_new"]["mw"],
                        EP_result["obj_new"]["vw"],
                        EP_result["incoming_mw"],
                        EP_result["incoming_vw"])
                #TODO if results would be wrong check strange transposition of matrix
                result = {
                    "mw": EP_result["outgoing_mw"],
                    "vw": EP_result["outgoing_vw"],
                    "iter": EP_result["iter"],
                }
                results[self.id] = result
                self.new_iteration = False
        print("Finish one of the data listeners")