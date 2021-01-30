import numpy as np
import random
from scipy.linalg import hadamard
from scipy import signal


def sign(x, delta):
    if x < 0: 
        return -delta
    else:
        return delta


def greedy_quantizer(Phi, image_list, delta):  # greedy quantizer (r=1)
    binary_list = []
    m = Phi.shape[0]
    for im_vec in image_list:
        y = Phi @ im_vec
        q = np.zeros(m)
        u = np.zeros(m)
        q[0] = sign(y[0], delta)
        u[0] = y[0] - q[0]
        for i in range(m-1):
            q[i+1] = sign(u[i] + y[i+1], delta)
            u[i+1] = u[i] + y[i+1] - q[i+1] 
        binary_list.append(q)
    return binary_list


def convolve(h, v, n):
    return sum([h[k]*v[n-k] for k in range(1,n+1)])


def first_quantizer(Phi, image_list, delta):   # first order Sigma-Delta quantizer (r=1)
    binary_list = []
    m = Phi.shape[0]
    h = signal.unit_impulse(m, idx=1)
    for im_vec in image_list:
        y = Phi @ im_vec
        q = np.zeros(m)
        v = np.zeros(m)
        for i in range(m):
            conv = convolve(h,v,i)
            q[i] = sign(conv + y[i], delta)
            v[i] = conv + y[i] - q[i]
        binary_list.append(q)
    return binary_list


def second_quantizer(Phi, image_list, delta):  # second order Sigma-Delta quantizer (r=2)
    binary_list = []
    m = Phi.shape[0]
    h = (7*signal.unit_impulse(m, idx=1)-signal.unit_impulse(m, idx=7))/6
    for im_vec in image_list:
        y = Phi @ im_vec
        q = np.zeros(m)
        v = np.zeros(m)
        for i in range(m):
            conv = convolve(h,v,i)
            q[i] = sign(conv + y[i], delta)
            v[i] = conv + y[i] - q[i]
        binary_list.append(q)
    return binary_list


def third_quantizer(Phi, image_list, delta):  # third order Sigma-Delta quantizer (r=3)
    binary_list = []
    m = Phi.shape[0]
    h = 175/144*signal.unit_impulse(m, idx=1)-25/108*signal.unit_impulse(m, idx=7)+7/432*signal.unit_impulse(m, idx=25)
    for im_vec in image_list:
        y = Phi @ im_vec
        q = np.zeros(m)
        v = np.zeros(m)
        for i in range(m):
            conv = convolve(h,v,i)
            q[i] = sign(conv + y[i], delta)
            v[i] = conv + y[i] - q[i]
        binary_list.append(q)
    return binary_list


def error_dist(V, image_list, binary_list):
    k = len(image_list)
    error_list = []
    error_list_abs = []
    count = 0
    for i in range(k):
        for j in range(i+1, k):
            dist1 = np.linalg.norm(image_list[i]-image_list[j])
            dist2 = np.linalg.norm(V @ (binary_list[i]-binary_list[j]), ord=1)
            if dist1 != 0:
                error_relative = (dist1-dist2)/dist1
                count += 1
            else:
                error_relative = 0
            error_list.append(abs(error_relative))
            error_list_abs.append(abs(dist1-dist2))
    return sum(error_list)/count, sum(error_list_abs)/count


def error_JL_dist(Phi, V, image_list):   # compared with Johnson–Lindenstrauss (JL) embedding
    k = len(image_list)
    error_list = []
    error_list_abs = []
    count = 0
    B = V @ Phi
    for i in range(k):
        for j in range(i+1, k):
            dist1 = np.linalg.norm(image_list[i]-image_list[j])
            dist2 = np.linalg.norm(B@(image_list[i]-image_list[j]), ord=1)
            if dist1 != 0:
                error_relative = (dist1-dist2)/dist1
                count += 1
            else:
                error_relative = 0
            error_list.append(abs(error_relative))
            error_list_abs.append(abs(dist1-dist2))
    return sum(error_list)/count, sum(error_list_abs)/count

