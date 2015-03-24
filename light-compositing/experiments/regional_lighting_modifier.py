#!/usr/bin/python2
import mdp
import numpy as np

import cv2
from matplotlib import pyplot as plt


def main():
    img_name = "../../../input_image/basket/images/%03d.png" % (000)
    img = cv2.imread(img_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    log_img = np.ones(img.shape, np.float32)

    np.log(img, log_img)
    log_img = np.nan_to_num(log_img)
    print log_img
    cv2.PCACompute(img)
    y = mdp.pca(log_img)
    print y
main()
