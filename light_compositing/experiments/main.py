#!/usr/bin/python2

import sys

import basis_lights_pixel
import cv2
sys.path.append("..")
import image_utils as utils






def main():
    img_list = []
    for i in range(0, 10):
        img_name = "../test_data/%03d.png" % (i)
        img = cv2.imread(img_name)

        # Normalize images to range [0..1] so that we can more easily
        # do calculations on them
        img_list.append(utils.normalize(img))

    res_image = basis_lights_pixel.edge_lights(img_list)
    print res_image
    cv2.imshow('image', res_image)
    cv2.waitKey(0)

    # cv2.imwrite('test.png', avg_image)
    cv2.imwrite('test.png', utils.denormalize(res_image))
    print(utils.denormalize(res_image))
main()
