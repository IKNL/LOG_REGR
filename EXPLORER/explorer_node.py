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
        # since we use only one row, I changed it to 1-dimensional array
        v = np.ones((n)) * math.inf

        vw = vp
        # mw dimensions are d * 1
        mw = mp
    else:
        a = obj["state"]["a"]
        m = obj["state"]["m"]
        v = obj["state"]["v"]

        vw = vp
        mw = mp

    iterations_number = 1
    is_last_iteration = False
    stability = np.zeros(n)
    zi = np.zeros(n)

    for iteration in range(0, iterations_number):
        is_last_iteration = is_last_iteration or (iteration == iterations_number - 1)
        nskip = 0
        old_mw = mw
        for i in range(0, n):
            # vw is size d * d
            vw = np.eye(d)
            # vwx is size d * 1
            # i is inside brackets to get a column vector
            vwx = np.matmul(vw, x[:, [i]])
            # xvwx is a number
            # i is not inside brackets since the form returned would be 1 * d
            # item is added to get a single value
            xvwx = np.matmul(x[:, i], vwx).item()

            if np.isfinite(v[i]):
                # v0 dimensions are d * d
                v0 = vw + np.matmul(vwx, vwx.T) * ((v[i] - xvwx) ** -1)
                # v0x dimensions are d * 1
                v0x = vwx * (v[i] / (v[i] - xvwx))
                # xv0x is a number
                xv0x = 1 / (1 / xvwx - 1 / v[i])
                # m0 dimensions are d * 1
                m0 = mw + v0x / (v[i] * (np.matmul(x[:, i].T, mw) - m[i]))
            else:
                v0 = vw
                v0x = vwx
                xv0x = xvwx
                m0 = mw
            if xv0x < 0:
                nskip += 1
                continue

            #  xm is a number
            # I put item at the end so method returns a single value instead of 1*1 array
            xm = np.matmul(x[:, i].T, m0).item()
            # z is a number
            z = xm / math.sqrt(math.pi / 8 * xv0x + 1)
            # true is a number
            true = 1 / (1 + math.exp(z))
            # alpha is a number
            alpha = true / math.sqrt(math.pi / 8 * xv0x + 1)

            # mw dimensions are d * 1
            mw = m0 + v0x * alpha
            # xmw is a number
            # item is added to get a single value
            xmw = np.matmul(x[:, i].T, mw).item()

            assert (z != np.nan)

            stability[i] = abs(xv0x)
            zi[i] = z
            # prev_v is a number
            prev_v = v[i]
            if restrict and v[i] < 0:
                v[i] = prev_v
            else:
                vw = v0 - np.matmul(v0x, v0x.T) * (alpha * (math.pi / 8 * xmw + alpha) / (math.pi / 8 * xv0x + 1))


X = np.array([[10, 20], [10, 20], [10, 20]])
Y = np.array([[-1], [1], [-1]])
# XTR, YTR, lik_method, prior_mean, prior_variance, state=None
train_linear(X, Y, "", [3], "")
