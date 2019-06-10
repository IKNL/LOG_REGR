from EXPLORER.explorer_node import train_linear
import numpy as np

new_iteration = True
close = False


def run_one_more_iteration():
    global new_iteration
    new_iteration = False


def stop_computations():
    global close
    close = True


def gaussian_division(m1, v1, m2, v2):
    v = np.linalg.inv(np.linalg.inv(v1) - np.linalg.inv(v2))
    # left division in pythonr returns 3 arguments, first of them is a solution equivalent to matlab
    m = np.matmul(v, np.linalg.lstsq(v1, m1)[0] - np.linalg.lstsq(v2, m2)[0])
    return m, v


# in the original data is stored inside the sockets, here connection is modified so data is accepted instead of
def new_job_listener(XTR, YTR, results, central_message):
    YTR[YTR == 0] = -1
    EP_result = {}
    EP_result["obj_new"] = {}
    EP_result["incoming_mw"] = []
    EP_result["incoming_vw"] = []
    EP_result["outgoing_mw"] = []
    EP_result["outgoing_vw"] = []
    EP_result["obj_new"]["state"] = []
    while not close:
        if new_iteration:
            EP_result["iter"] = central_message["iter"]
            EP_result["obj"] = EP_result["obj_new"]
            EP_result["obj_new"] = train_linear(XTR, YTR, "logistic", central_message["mw"], central_message["vw"],
                                                EP_result)
            if EP_result["iter"] == 1:
                EP_result["outgoing_mw"] = EP_result["obj_new"]["mw"]
                EP_result["outgoing_vw"] = EP_result["obj_new"]["vw"]
            else:
                EP_result.incoming_mw, EP_result.incoming_vw = gaussian_division(
                    central_message["mw"],
                    central_message["vw"],
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
                "iter": EP_result["iter"]
            }
            results.append(result)
    print("Finish one of the data listeners")