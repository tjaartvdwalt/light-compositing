#!/usr/bin/env python2
import sys

import numpy as np
from scipy.optimize import minimize

import cv2
import image_utils as utils
from my_gradient_map import gradient_map

img_list = []
N = 30
gmap = None
weight = 1


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


def sum_images(l, images):
    # channels = img_list[0].shape[2]
    return_image = np.zeros_like(images[0])
    # sum = np.zeros(channels)

    for i, img in enumerate(images):
        return_image += np.multiply(l[i], img)

    return return_image


def edge_light(l):
    global gmap
    global weight

    rows = img_list[0].shape[0]
    cols = img_list[0].shape[1]

    res = 0
    for r in range(rows):
        for c in range(cols):
            # print "sum_lambda %s" % sum_lambda_int(l, r, c)
            # print "gradient map %s" % (gradient_map**2)
            # diff = np.gradient(sum_lambda_int(l, r, c))
            # print diff
            # gradient = np.arctan(diff[1]/diff[0]) * 180 / np.pi
            gradient = sum_lambda_int(l, r, c)
            res += weight.item(r, c) * np.linalg.norm(gradient -
                                                      (gmap.item(r, c)**2))**2
    print "lambda: %s" % l
    print "sum:    %s" % res
    return res


def main():
    global img_list
    global gmap
    global weight

    img_list = utils.read_images("../../test_data/cafe", N, downsample=3)
    full_img_list = utils.read_images("../../test_data/cafe", N)
    gray_imgs = utils.read_images("../../test_data/cafe", N, gray=True)
    x0 = np.full(N, 1.0/N)

    (gmap, weight) = gradient_map(gray_imgs)

    bnds = []
    for i in range(len(img_list)):
        bnds.append((0, 1))

    lambdas = minimize(edge_light, x0, method='TNC', jac=False,
                       bounds=bnds)

    ret_image = sum_images(lambdas.x, full_img_list)
    print lambdas.message
    print "Choice of lambdas = %s" % (lambdas.x)

    cv2.imwrite('output_edge.png', utils.denormalize_img(ret_image))
    cv2.imshow('image', ret_image)
    cv2.waitKey(0)

main()
