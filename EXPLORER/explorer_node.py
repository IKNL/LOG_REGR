import numpy as np
import numpy.matlib
import math


def bpm_ep(X, Y, type, prior_mean, prior_variance, state):
    # since logistic regression, e = 1
    e = 1
    X = np.c_[np.ones((X.shape[0], 1)), X]

    # np.multiple - element-wise multiplication in python
    data = np.multiply(X, np.matlib.repmat(Y, 1, X.shape[1]))
    obj = {"type": type, "e": e, "add_bias": 1, "mp": [], "vp": [], "s": [], "mw": [], "vw": [], "alpha": [],
           "bias": [],
           "X": [], "Y": [], "state": state, "restrict": 0, "stepsize": 1, "train_err": [], "loo": [], "loo_count": [],
           "stability": [],
           "data": data, "prior_variance": prior_variance, "prior_mean": prior_mean}
    return obj


def train_linear(XTR, YTR, lik_method, prior_mean, prior_variance, state=None):
    if state == None:
        state = []
    obj = bpm_ep(XTR, YTR, lik_method, prior_mean, prior_variance, state)
    x = obj["data"].T
    e = obj["e"]
    restrict = obj["restrict"]
    type = obj["type"]
    d, n = x.shape
    if not obj["mp"]:
        if len(obj["prior_mean"]) == 1:
            mp = np.ones((d, 1)) * obj["prior_mean"][0]
        else:
            mp = obj["prior_mean"]
            # assert (len(obj["prior_mean"]) == d)
    else:
        mp = obj["mp"]

    if not obj["vp"]:
        if len(obj["prior_variance"]) == 1:
            vp = np.eye(d) * obj["prior_variance"]
        else:
            vp = obj["prior_variance"]
            # assert (len(np.diagonal(obj["prior_variance"])) == d)
    else:
        vp = obj["vp"]

    if not obj["state"]:

        a = np.zeros((1, n))
        m = np.ones((1, n))
        v = np.ones((1, n)) * math.inf

        vw = vp
        mw = mp
    else:
        a = obj["state"]["a"]
        m = obj["state"]["m"]
        v = obj["state"]["v"]

        vw = vp
        mw = mp

    iterations_number = 1
    is_last_iteration = False

    for iteration in range(0, iterations_number):
        is_last_iteration = is_last_iteration or (iteration == iterations_number - 1)
        nskip = 0
        old_mw = mw
        for i in range(0, n - 1):
            vw = np.eye(d)
            vwx = np.matmul(vw, x[:, i])
            xvwx = np.matmul(x[:, 1].T, vwx)
            if np.isfinite(v[i]):

            else:


X = np.array([[10, 20], [10, 20], [10, 20]])
Y = np.array([[-1], [1], [-1]])
train_linear(X, Y, "", "", "")
