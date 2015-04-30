#!/usr/bin/env python2
import sys

import numpy as np

import cv2
sys.path.append("..")
import image_utils as utils

img_list = []
N = 15


def get_mixture_coefficients(sigma=0.5):
    val = 1.0/N
    i_coef = np.transpose(np.full(N, val))
    # print val
    # print i_coef
    s = S(sigma)
    print s
    s_times_lambda = np.dot(np.matrix(s), i_coef)
    # print np.matrixmultiply(np.matrix(s), i_coef)
    # for i in range(10):
    n_coef = (np.linalg.norm(i_coef) /
              np.linalg.norm(s_times_lambda)) * s_times_lambda
        # print(i_coef)

    return n_coef


def S(sigma):
    soft_lambda = np.ones((N, N), dtype="float32")
    print sigma
    for i in range(N):
        for j in range(N):
            soft_lambda.itemset(i, j, element_s(img_list[i],
                                                img_list[j], sigma))
    return soft_lambda


def element_s(i, j, sigma):
    # print i - j
    # print j
    return np.exp(-(np.linalg.norm(i - j)**2)/(2 * (sigma**2)))


def sum_lambda_int(l):
    # channels = img_list[0].shape[2]
    return_image = np.zeros_like(img_list[0])
    # sum = np.zeros(channels)
    i = 0
    for img in img_list:
        return_image += np.multiply(l.item(0, i), img)
        i += 1

    return return_image


def main():
    # px1 = np.array([0, 126, 255], dtype=float)
    # px2 = np.array([0, 0, 0], dtype=float)
    for i in range(0, N):
        img_name = "../test_data/basket/%03d.png" % (i)
        print(img_name)
        img = cv2.imread(img_name)
        # img_list.append(utils.normalize(img))
        img_list.append(img)

    coef = get_mixture_coefficients(sigma=10)

    soft_light_img = sum_lambda_int(coef)

    hsv_image = cv2.cvtColor(soft_light_img, cv2.cv.CV_BGR2HSV)

    for i in range(hsv_image.shape[0]):
        hsv_image[i][:, 2] = hsv_image[i][:, 2] + 6

    # hsv_image = np.clip(hsv_image[:, 2], 0.0, 1.0)
    # print hsv_image[:, 2]
    eq_soft_light_img = cv2.cvtColor(hsv_image, cv2.cv.CV_HSV2BGR)

    # global_resize = [resize_factor, resize_factor,
    #                  resize_factor]
    # eq_soft_light_img = np.add(soft_light_img, global_resize)
    
    # (b, g, r) = cv2.split(utils.denormalize(soft_light_img))
    # eq_b = cv2.equalizeHist(b)
    # eq_g = cv2.equalizeHist(g)
    # eq_r = cv2.equalizeHist(r)

    # eq_soft_light_img = cv2.merge((eq_b, eq_g, eq_r))
    cv2.imwrite('output_soft.png', eq_soft_light_img)
    cv2.imshow('image', eq_soft_light_img)
    cv2.waitKey(0)


    # print coef
    # print alpha(10, 10)
    # print px_avg(400, 500)
    # print px_avg(img_list, 0, 0)
    # px1 = np.ndarray(shape=(3, 1), dtype=float,
    # buffer=np.array([0, 126, 255]))
    # px2 = np.ndarray(shape=(1, 3), dtype=float,
    #                  buffer=np.array([[255], [0], [128]]))
    # diffuse_color(1)

main()
