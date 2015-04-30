#!/usr/bin/env python2
import numpy as np
from scipy.optimize import minimize

import cv2

import sys
sys.path.append("..")
import image_utils as utils
import my_gradient_map

N = 10
img_list = []


def main():
    for i in range(0, N):
        img_name = "../../../input_image/basket/images/%03d.png" % (i)
        # img_name = "../test_data/small/%03d_small.png" % (i)
        img = cv2.imread(img_name)
        gray_image = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
        img_list.append(utils.normalize(gray_image))

    gradient_map = my_gradient_map.gradient_map(img_list)

    cv2.imwrite('output_diffuse.png', utils.denormalize(gradient_map))
    cv2.imshow('image', gradient_map)
    cv2.waitKey(0)

main()
