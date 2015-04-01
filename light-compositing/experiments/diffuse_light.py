#!/usr/bin/env python2
import math
import sys

import numpy as np
from scipy.optimize import minimize

import cv2
sys.path.append("..")
import image_utils as utils

N = 10
W = np.array([1, 1, 1])
SIGMA = 0.5
EPSILON = 0.01
img_list = []


def angle(px1, px2):
    # print np.linalg.norm(px1, ord=None)
    # print np.linalg.norm(px2)

    with np.errstate(divide='ignore'):
        t1 = px1 / np.linalg.norm(px1)
        # this check converts answer to 0 if we divided by 0
        t1[np.isnan(t1)] = 0
        t2 = px2 / np.linalg.norm(px2)
        t2[np.isnan(t2)] = 0

    return np.arccos(np.dot(t1, t2))
    # print(t3)


def alpha(r, c):
    # print "angle %s" % math.exp(-(angle(px_avg(r, c), W)**2)/(2*SIGMA**2))
    return math.exp(-(angle(px_avg(r, c), W)**2)/(2*SIGMA**2))


def px_avg(r, c):
    channels = img_list[0].shape[2]
    sum = np.zeros(channels)

    for img in img_list:
        px = img[r, c]
        sum += px

    avg = np.ones(channels)
    avg *= len(img_list)
    # print(len(img_list))
    return sum / avg


def sum_lambda_int(l, r, c):
    channels = img_list[0].shape[2]
    sum = np.zeros(channels)
    i = 0
    for img in img_list:
        px = img[r, c]
        sum = l[i] * px
        i += 1
    return sum


def w_hat(l, r, c):
    return sum_lambda_int(l, r, c) / sum_lambda_int(l, r, c) + EPSILON


def diffuse_color(l):
    img_list
    rows = img_list[0].shape[0]
    cols = img_list[0].shape[1]

    for r in range(rows):
        for c in range(cols):
            alpha(r, c) * angle(sum_lambda_int(l, r, c), px_avg(r, c))
            - (1 - alpha(r, c)) * w_hat(l, r, c) * angle(
                sum_lambda_int(l, r, c), W)


def main():
    # px1 = np.array([0, 126, 255], dtype=float)
    # px2 = np.array([0, 0, 0], dtype=float)

    for i in range(0, N):
        img_name = "../test_data/small/%03d_small.png" % (i)
        img = cv2.imread(img_name)
        img_list.append(utils.normalize(img))

    # print alpha(10, 10)
    # print px_avg(400, 500)
    # print px_avg(img_list, 0, 0)
    # px1 = np.ndarray(shape=(3, 1), dtype=float,
    # buffer=np.array([0, 126, 255]))
    # px2 = np.ndarray(shape=(1, 3), dtype=float,
    #                  buffer=np.array([[255], [0], [128]]))
    # diffuse_color(1)
    x0 = np.ones(N)
    res = minimize(diffuse_color, x0, method='Nelder-Mead')
    print("lambda: %s" % res)
main()
