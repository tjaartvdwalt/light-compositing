#!/usr/bin/python2
import math

import numpy as np

import cv2


def orientation_map(mag, ori, threshold=0.01):
    # print(mag.shape)
    return_image = np.zeros_like(mag)
    rows = mag.shape[0]
    cols = mag.shape[1]
    # channels = mag.shape[2]

    for r in range(rows):
        for c in range(cols):
            # for channel in range(channels):
            if mag.item(r, c) > threshold:
                return_image.itemset(r, c,
                                     ori.item(r, c))
    return return_image


def calcHist(oris, mags):
    bins = []
    for i in range(37):
        bins.append([])

    for i, ori in enumerate(oris):
        if(ori > 180):
            val = ori - 180
        else:
            val = ori
        index = int(val / 5)
        bins[index].append((ori, mags[i]))
    return bins


def get_max_index(bins):
    max_value = 0
    max_index = 0

    for i, bin in enumerate(bins):
        if len(bin) > max_value:
            max_value = len(bin)
            max_index = i

    return max_index


def get_max_weight(bins):
    max_index = get_max_index(bins)
    max_value = len(bins[max_index])

    total = 0
    for i, bin in enumerate(bins):
        total += len(bin)

    return 1.0 * max_value / total


def get_max_mag_index(my_bin):
    max_value = 0
    max_index = 0
    for i, value in enumerate(my_bin):
        if value > max_value:
            max_value = value
            max_index = i

    return max_index


def calculate_gradient_map(gradient_maps):
    shape = gradient_maps[0][0].shape
    rows = shape[0]
    cols = shape[1]
    return_map = np.zeros((rows, cols))
    weight_map = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            ori_at_images = np.zeros((len(gradient_maps)), dtype=np.float32)
            mag_at_images = np.zeros((len(gradient_maps)), dtype=np.float32)
            for i, im in enumerate(gradient_maps[0]):
                ori_at_images.itemset(i, im[r][c])

            for i, im in enumerate(gradient_maps[1]):
                mag_at_images.itemset(i, im[r][c])

            hist = calcHist(ori_at_images, mag_at_images)
            index = get_max_index(hist)
            weight_map.itemset(r, c, get_max_weight(hist))
            max_mag_index = get_max_mag_index(hist[index])
            return_map.itemset(r, c, hist[index][max_mag_index][0])

    return (return_map, weight_map)


def gradient_map(images):
    gmaps = []
    for i, image in enumerate(images):
        sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=-1)
        sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=-1)

        mag = cv2.magnitude(sx, sy)
        ori = cv2.phase(sx, sy, angleInDegrees=True)

        # cv2.imshow('image', ori)
        # cv2.waitKey(0)
        # cv2.imshow('orientation', ori)
        # cv2.waitKey(0)
        # cv2.imshow('magnitude', mag)
        # cv2.waitKey(0)

        # ori_map = orientation_map(mag, ori)
        # # norm_ori = ori / 360.0
        # # norm_ori = ori * 255
        gmaps.append((ori, mag))
        # # print(ori_map)

    (gmap, weight) = calculate_gradient_map(gmaps)
    return (gmap, weight)
