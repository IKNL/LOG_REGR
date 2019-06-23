import numpy as np
import numpy.matlib
import math
from scipy.stats import norm
from scipy.misc import logsumexp


def gaussian_division(m1, v1, m2, v2):
    v = np.linalg.inv(np.linalg.inv(v1) - np.linalg.inv(v2))
    # left division in pythonr returns 3 arguments, first of them is a solution equivalent to matlab
    m = np.matmul(v, np.linalg.lstsq(v1, m1)[0] - np.linalg.lstsq(v2, m2)[0])
    return m, v


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
        state = {}
    obj = bpm_ep(XTR, YTR, lik_method, prior_mean, prior_variance, state)
    x = obj["data"].T
    show_progress = True
    e = obj["e"]
    restrict = obj["restrict"]
    type = obj["type"]
    d, n = x.shape
    if not obj["mp"]:
        if len(obj["prior_mean"]) == 1:
            # mp form is d * 1
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

        # since we use only one row, I changed it to 1-dimensional array
        a = np.zeros(n)
        # since we use only one row, I changed it to 1-dimensional array
        m = np.ones(n)
        # since we use only one row, I changed it to 1-dimensional array
        v = np.ones(n) * math.inf

        vw = vp
        # mw dimensions are d * 1
        mw = mp
    else:
        a = obj["state"]["a"]
        m = obj["state"]["m"]
        v = obj["state"]["v"]

        vw = vp
        mw = mp

    iterations_number = 10
    is_last_iteration = False
    stability = np.zeros(n)
    zi = np.zeros(n)

    for current_iteration in range(0, iterations_number):
        is_last_iteration = is_last_iteration or (current_iteration == iterations_number - 1)
        nskip = 0
        old_mw = mw
        for data_row in range(0, n):
            # vw is size d * d
            vw = np.eye(d)
            # vwx is size d * 1
            # i is inside brackets to get a column vector
            vwx = np.matmul(vw, x[:, [data_row]])
            # xvwx is a number
            # i is not inside brackets since the form returned would be 1 * d
            # item is added to get a single value
            xvwx = np.matmul(x[:, data_row], vwx).item()

            if np.isfinite(v[data_row]):
                # v0 dimensions are d * d
                v0 = vw + np.matmul(vwx, vwx.T) * ((v[data_row] - xvwx) ** -1)
                # v0x dimensions are d * 1
                v0x = vwx * (v[data_row] / (v[data_row] - xvwx))
                # xv0x is a number
                xv0x = 1 / (1 / xvwx - 1 / v[data_row])
                # m0 dimensions are d * 1
                m0 = mw + v0x / (v[data_row] * (np.matmul(x[:, data_row].T, mw) - m[data_row]))
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
            xm = np.matmul(x[:, data_row].T, m0).item()
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
            xmw = np.matmul(x[:, data_row].T, mw).item()

            assert (z != np.nan)

            # generalization error estimate
            stability[data_row] = abs(xv0x)
            zi[data_row] = z

            # prev_v is a number
            prev_v = v[data_row]
            # function for logistic regression is taken
            v[data_row] = (math.pi / 8 * xv0x + 1) / (alpha * (math.pi / 8 * xmw + alpha * 1)) - xv0x
            if restrict and v[data_row] < 0:
                v[data_row] = prev_v
            else:
                # TODO experiment with if 0 condition
                # ADF update of vw for logistic regression
                vw = v0 - np.matmul(v0x, v0x.T) * (alpha * (math.pi / 8 * xmw + alpha) / (math.pi / 8 * xv0x + 1))

            m[data_row] = xm + (xv0x + v[data_row]) * alpha

            if show_progress or is_last_iteration:
                # p is a number
                p = -0.5 * alpha * (math.pi / 8 * xv0x + 1) / (math.pi / 8 * xmw + alpha)
                p = p - 0.5 * np.log(1 + xv0x / v[data_row])
                a[data_row] = true - p

        ev = np.ones(iterations_number)
        if nskip > 0:
            #TODO move printing to the file
            print('skipped {} points on iter {}\n'.format(nskip, iter))
        if show_progress:
            #TODO remove counting of s in the end if it is already calculated here
            s = np.matmul(np.matmul(mp.T, np.linalg.inv(vp)), mp) - np.matmul(np.matmul(mw.T, np.linalg.inv(vw)), mw)
            for data_row in range(0, n):
                s = s + (m[data_row] ** 2) / v[data_row]
                # slodget in python provides sign and log-determinant so the later only is selected
            ev[current_iteration] = 0.5 * np.linalg.slogdet(vw)[1] - 1 / 2 * s + sum(a) - 0.5 * np.linalg.slogdet(vp)[1]

        #TODO add plotting figures as in MATLAB code

        if max(abs(mw - old_mw)) < 1e-8:
            if is_last_iteration:
                break
            else:
                is_last_iteration = True
    if current_iteration == iterations_number - 1:
        print('not enough iterations')
    else:
        print('EP converged in %d iterations\n'.format(iter))

    s = np.matmul(np.matmul(mp.T, np.linalg.inv(vp)), mp) - np.matmul(np.matmul(mw.T, np.linalg.inv(vw)), mw)
    for data_row in range(0, n):
        s = s + (m[data_row] ** 2) / v[data_row]

    s = 0.5 * np.linalg.slogdet(vw)[1] - 1 / 2 * s + sum(a) - 0.5 * np.linalg.slogdet(vp)[1]

    if 1:
        # generalization error estimates
        obj["stability"] = np.mean(stability)
        obj["loo"] = np.exp(logsumexp(norm.cdf(-25 * zi))) / n
        negative_zi = [zp for zp in zi if zp <= 0]
        obj["loo_count"] = np.mean(negative_zi)
        negative_mwx = [el for el in np.matmul(x, mw) if el <= 0]
        obj["train_err"] = np.mean(negative_mwx)

    obj["s"] = s
    obj["mw"] = mw
    obj["vw"] = vw
    obj["state"]["a"] = a
    obj["state"]["m"] = m
    obj["state"]["v"] = v
    return obj

