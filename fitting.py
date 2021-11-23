import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


def rm_rep(curve):
    """ Remove all the repetitive points in a sampled curve from the sketch
        curve:    Sampled data points. Array with the size [num, dim]
    """
    cv = curve.copy()
    pts = [cv[0, :]]
    for ii in range(1, cv.shape[0]):
        if not (cv[ii, :] == cv[ii - 1, :]).all():
            pts.append(cv[ii, :])
    pts = np.array(pts)
    if cv.shape[0] - pts.shape[0] != 0:
        print("Delete %d repetitive point(s)." % (cv.shape[0] - pts.shape[0]))
    return pts


def bsf(curve, m, closed=False, plot=False):
    """ Fit a B-spline curve to the sampled points with given number of
        control points.
        curve:   Sampled data points. Array with the size [num, dim]
        m:       Desired number of  control points.
        closed:  Closed curves would have a repetitive data point
                 at the end. Set to true to implicate this.
        plot:    Whether to plot it out or not.
    """
    # Record the start time
    start = time.time()
    pts = curve.copy()
    pts = rm_rep(pts)
    # Construct the knot vector given number of control points
    t3 = np.zeros(m + 4)
    t3[3:-3] = np.linspace(0, 1, num=(m - 2))
    t3[-3:] = np.ones(3)
    t3 = t3.tolist()
    # Fit the B-Spline Curve given desired number of control points
    # Ignore the fitting warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The required storage space exceeds the available storage space.")
        if closed:
            tck, u, _, _, _ = interpolate.splprep(pts.T, u=None, t=t3, nest=m+4, s=1, k=3, per=1)
        else:
            tck, u, _, _, _ = interpolate.splprep(pts.T, u=None, t=t3, nest=m+4, s=1, k=3)
    cof = tck[1]
    print('Obtain %d control points.' % cof[0].shape[0])
    xx = np.linspace(u.min(initial=np.infty), u.max(initial=-np.infty), curve.shape[0])
    spl = interpolate.splev(xx, tck)
    mse = (np.square(np.array(spl).T - curve)).mean(axis=0)
    print('The mean square error is: ', mse, '.')
    # Calculate the time cost
    end = time.time()
    print('The process takes %f seconds' % (end-start))
    if plot:
        # Draw out the fit spline with computed control points
        fig, ax = plt.subplots(figsize=(7, 4), dpi=125)
        ax.plot(curve[:, 0], curve[:, 1], "r", label='Original Data Points')
        ax.plot(spl[0], spl[1], 'g', label='Fitted B-Spline Curve')
        ax.plot(cof[0], cof[1], '.c', markersize=12, label='Control Points')
        ax.plot(cof[0], cof[1], '--c')
        plt.grid()
        plt.legend(loc='best')
        plt.show()
    return pts, tck, u
