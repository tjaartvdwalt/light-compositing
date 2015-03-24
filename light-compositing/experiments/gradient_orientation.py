#!/usr/bin/python2
import numpy as np

import cv2
from matplotlib import pyplot as plt


def orientation_map(mag, ori, threshold=0.01):
    print(mag.shape)
    return_image = np.zeros(mag.shape, np.float32)
    rows = mag.shape[0]
    cols = mag.shape[1]

    # channels = mag.shape[2]

    for r in range(rows):
        for c in range(cols):
            # for channel in range(channels):
            if mag.item(r, c) > threshold:
                return_image.itemset(r, c, ori.item(r, c))
    return return_image


def calculate_gradient_histograms(gradient_maps):
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
            print hist
            plt.plot(hist)
            plt.show()
            # print(pixel_at_images)


def main():
    gradient_maps = []
    for i in range(0, 10):
        img_name = "../../../input_image/basket/images/%03d.png" % (i)
        img = cv2.imread(img_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=-1)
        sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=-1)

        mag = cv2.magnitude(sx, sy)
        ori = cv2.phase(sx, sy, angleInDegrees=True)

        ori_map = orientation_map(mag, ori)
        # norm_ori = ori / 360.0
        # norm_ori = ori * 255
        gradient_maps.append(ori_map)
        print(ori_map)
        cv2.imshow('image', ori_map)
        cv2.waitKey(0)

    calculate_gradient_histograms(gradient_maps)

main()
