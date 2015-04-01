#!/usr/bin/python2
import numpy as np

import cv2
from matplotlib import pyplot as plt


def orientation_map(mag, ori, threshold=0.01):
    # print(mag.shape)
    return_image = np.zeros(mag.shape, np.float32)
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


def calculate_gradient_histograms(gradient_maps):
    return_map = np.zeros(gradient_maps[0].shape, np.float32)
    shape = gradient_maps[0].shape
    rows = shape[0]
    cols = shape[1]

    for r in range(rows):
        for c in range(cols):
            i = 0
            pixel_at_images = np.zeros((len(gradient_maps)), np.float32)
            for im in gradient_maps:
                pixel_at_images.itemset(i, im.item(r, c))
                i += 1
            hist = cv2.calcHist([pixel_at_images], [0], None, [72], [0, 360])

            max = 0
            i = 0
            index = 0
            # print pixel_at_images
            for val in hist:
                if val > max:
                    max = val
                    index = i
                i += 1

            cur_val = 0
            lower_bound = index * 5
            upper_bound = (index + 1) * 5

            for pixel in pixel_at_images:
                if (pixel >= lower_bound) and (pixel < upper_bound):
                    if pixel > cur_val:
                        cur_val = pixel
            # print "index of max value: %s, max val: %s " % (index, max)
            # print "item (%s, %s) = %s" %(r, c, cur_val)
            return_map.itemset(r, c, cur_val)
            # print hist
            # plt.plot(hist)
            # plt.show()
            # print(pixel_at_images)
    return return_map


def gradient_map(images):
    gradient_maps = []
    for i in range(0, len(images)):

        # convert images to grayscale
        if len(images[0].shape) > 2:
            gray_image = cv2.cvtColor(images[i], cv2.cv.CV_BGR2GRAY)
        else:
            gray_image = images[i]

        sx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=-1)
        sy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=-1)

        mag = cv2.magnitude(sx, sy)
        ori = cv2.phase(sx, sy, angleInDegrees=True)

        # cv2.imshow('image', ori)
        # cv2.waitKey(0)

        ori_map = orientation_map(mag, ori)
        # norm_ori = ori / 360.0
        # norm_ori = ori * 255
        gradient_maps.append(ori_map)
        # print(ori_map)
        # cv2.imshow('image', ori_map)
        # cv2.waitKey(0)

    return calculate_gradient_histograms(gradient_maps)
