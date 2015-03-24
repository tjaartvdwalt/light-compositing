#!/usr/bin/env python2
import cv2
import numpy as np
import sys
sys.path.append("..")
import image_utils as utils


def angle(px1, px2):
    print np.linalg.norm(px1, ord=None)
    print np.linalg.norm(px2)

    t1 = px1 / np.linalg.norm(px1)
    # this check converts answer to 0 if we divided by 0
    t1[np.isnan(t1)] = 0
    t2 = px2 / np.linalg.norm(px2)
    t2[np.isnan(t2)] = 0

    print(t1)
    print(t2)
    t3 = np.arccos(np.dot(t1, t2))
    print(t3)


def alpa(px):
    pass


def px_avg(img_list, r, c):
    img_shape = img_list[0].shape
    sum = np.zeros(img_shape)

    for img in img_list:
        px = img[r, c]
        sum += px

    avg = np.ones(img_shape)
    avg *= len(img_list)
    avg = sum / avg

    return avg


def main():
    px1 = np.array([0, 126, 255], dtype=float)
    px2 = np.array([0, 0, 0], dtype=float)

    img_list = []

    for i in range(0, 10):
        img_name = "../test_data/%03d.png" % (i)
        img = cv2.imread(img_name)
        img_list.append(utils.normalize(img))

    px_avg(img_list, 0, 0)
    # px1 = np.ndarray(shape=(3, 1), dtype=float,
    # buffer=np.array([0, 126, 255]))
    # px2 = np.ndarray(shape=(1, 3), dtype=float,
    #                  buffer=np.array([[255], [0], [128]]))
    angle(px1, px2)
    alpha(px1)
main()
