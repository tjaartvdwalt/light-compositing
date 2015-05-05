#!/usr/bin/env python2
import math

import numpy as np
from scipy.optimize import minimize

import cv2
import image_utils as utils

# import pyipopt

# from scipy.optimize import leastsq

N = 10
W = np.array([1, 1, 1])
SIGMA = 0.5
EPSILON = 0.01
img_list = []


def angle(px1, px2):
    # print np.linalg.norm(px1, ord=None)
    # print np.linalg.norm(px2)

    if (px1[0] == 0) and (px1[1] == 0) and (px1[2] == 0):
        t1 = np.zeros(3)
    else:
        t1 = px1 / np.linalg.norm(px1)

    if px2[0] == 0 and px2[1] == 0 and px2[2] == 0:
        t2 = np.zeros(3)
    else:
        t2 = px2 / np.linalg.norm(px2)

    dot = np.dot(t1, t2)
    if dot > 1:
        dot = 1
    elif dot < -1:
        dot = -1

    return math.acos(dot)


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


def sum_images(l, images):
    # channels = img_list[0].shape[2]
    return_image = np.zeros_like(images[0])
    # sum = np.zeros(channels)

    for i, img in enumerate(images):
        return_image += np.multiply(l[i], img)

    return return_image


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


def w_hat(l, r, c):
    return sum_lambda_int(l, r, c) / (sum_lambda_int(l, r, c) + EPSILON)


def diffuse_color(l):
    # print(img_list[0])
    rows = img_list[0].shape[0]
    cols = img_list[0].shape[1]

    print img_list[0].shape
    sum = 0
    for r in range(rows):
        for c in range(cols):
            sum_lambda = sum_lambda_int(l, r, c)
            sum += alpha(r, c) * angle(sum_lambda, px_avg(r, c))
            - (1 - alpha(r, c)) * w_hat(l, r, c) * angle(sum_lambda, W)
    print "lambda: %s" % l
    print "sum:    %s" % sum
    return sum


def fprime(x):
    ones = np.ones_like(x)
    y = np.subtract(x, ones)

    # return (x - y)/np.
    # dx = np.diff(x)
    # dx = np.gradient(x)
    # y = diffuse_color(x)
    # return np.gradient(y, dx, edge_order=2)
    # next_x = x - 0.1

    # y = diffuse_color(x)
    # next_y = diffuse_color(next_x)
    # df = np.divide(y - next_y, x - next_x)
    # print "x      %s: " % x
    # print "next_x %s: " % next_x
    # print "y      %s: " % y
    # print "next_y %s: " % next_y
    # print "df     %s: " % df
    return dx


def main():
    # px1 = np.array([0, 126, 255], dtype=float)
    # px2 = np.array([0, 0, 0], dtype=float)
    global img_list
    img_list = utils.read_images("../../test_data/cafe", N, downsample=4)

    full_img_list = utils.read_images("../../test_data/cafe", N)

    # for i in range(0, N):
    #     # img_name = "../../../input_image/cafe/images/%03d.png" % (i)
    #     img_name = "../test_data/cafe/%03d.png" % (i)
    #     img = cv2.imread(img_name)
    #     img_list.append(utils.normalize_img(img))
    #     # img_list.append(img)

    # print alpha(10, 10)
    # print px_avg(400, 500)
    # print px_avg(img_list, 0, 0)
    # px1 = np.ndarray(shape=(3, 1), dtype=float,
    # buffer=np.array([0, 126, 255]))
    # px2 = np.ndarray(shape=(1, 3), dtype=float,
    #                  buffer=np.array([[255], [0], [128]]))
    # diffuse_color(1)

    x0 = np.full(N, 1.0/N)
    # print fprime(x0)
    # print x0
    # print diffuse_color(x0)
    # x0 = np.zeros(N)
    # print diffuse_color(x1)

    # x0 = np.ones(N)
    # print diffuse_color(x2)

    
    # results = pyipopt.fmin_unconstrained(diffuse_color, x0, fprime)

    # print results

    # nlp = pyipopt.create(diffuse_color, N)
    # x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)

    # nlp.close()

    # leastsq(diffuse_color, x0)

    bnds = []
    for i in range(len(img_list)):
        bnds.append((0, 1))


    lambdas = minimize(diffuse_color, x0, method='TNC', jac=False,
                       bounds=bnds)

    # lambdas = np.array([ 0.0772488 ,  0.07700344,  0.0769977 ,  0.07699107,  0.07702644,
    #     0.07702589,  0.07702531,  0.07694187,  0.07695671,  0.07701653,
    #     0.07696243,  0.07767547,  0.07693373])

    # lambdas = np.array([ 0.03336186,  0.03335751,  0.03336186,  0.03336186,  0.03335691,
    #     0.03336186,  0.03335623,  0.03335545,  0.03336186,  0.03316012,
    #     0.03336186,  0.03332896,  0.03336186,  0.03336186,  0.03335172,
    #     0.03335502,  0.03336186,  0.03336186,  0.03335585,  0.03333409,
    #     0.03335658,  0.03335828,  0.03335722,  0.03335778,  0.03336186,
    #     0.03335804,  0.03336186,  0.03335251,  0.03332677,  0.0333048 ])
    # print("lambda: %s" % lambdas)
    ret_image = sum_images(lambdas.x, full_img_list)
    print lambdas.message
    print "Choice of lambdas = %s" % (lambdas.x)
    cv2.imwrite('output_diffuse.png', utils.denormalize_img(ret_image))
    cv2.imshow('image', ret_image)
    cv2.waitKey(0)


main()
