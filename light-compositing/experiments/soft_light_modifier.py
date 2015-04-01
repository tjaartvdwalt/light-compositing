#!/usr/bin/env python2
import sys

import numpy as np

import cv2
sys.path.append("..")
import image_utils as utils

img_list = []

def S(i, j):
    Sij= exp(−kIi− Ijk2/2σs2)

def main():
    # px1 = np.array([0, 126, 255], dtype=float)
    # px2 = np.array([0, 0, 0], dtype=float)
    N = 10
    for i in range(0, N):
        img_name = "../test_data/%03d.png" % (i)
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
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    res = minimize(diffuse_color, 1, method='SLSQP')
    print("lambda: %s" % res)
main()
