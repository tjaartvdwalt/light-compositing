#!/usr/bin/env python2
import sys

import numpy as np
from scipy.optimize import minimize

import cv2
import my_gradient_map
sys.path.append("..")
import image_utils as utils

img_list = []
N = 10


# TODO: This function is the same as the one in diffuse light
def sum_lambda_int(l, r, c):
    # channels = img_list[0].shape[2]
    sum = 0
    # sum = np.zeros(channels)
    i = 0
    for img in img_list:
        px = img[r, c]
        sum = l[i] * px
        i += 1

    return sum


def weight():
    return 1


def arg_min(l):
    rows = img_list[0].shape[0]
    cols = img_list[0].shape[1]

    gradient_map = my_gradient_map.gradient_map(img_list)

    for r in range(rows):
        for c in range(cols):
            # print "sum_lambda %s" % sum_lambda_int(l, r, c)
            # print "gradient map %s" % (gradient_map**2)
            # diff = np.gradient(sum_lambda_int(l, r, c))
            # print diff
            # gradient = np.arctan(diff[1]/diff[0]) * 180 / np.pi
            gradient = sum_lambda_int(l, r, c)
            res = weight() * np.linalg.norm(gradient - (gradient_map.item(r, c)**2))**2
    return res


def edge_light(l):
    for i in range(N):
        l[i] * img_list[i]


def main():
    for i in range(0, N):
        img_name = "../../../input_image/basket/images/%03d.png" % (i)
        # img_name = "../test_data/small/%03d_small.png" % (i)
        img = cv2.imread(img_name)
        gray_image = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
        img_list.append(utils.normalize(gray_image))

    x0 = np.ones(N + 1)
    print x0
    # print x0[0]
    # print x0[0] * img_list[0]
    res = minimize(arg_min, x0, method='SLSQP')
    print res
    edge_light(res)

main()
