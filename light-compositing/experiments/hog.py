#!/usr/bin/python2
import cv2
import sys
sys.path.append("..")
import image_utils as utils
from skimage.feature import hog


def main():
    img_name = "astronaut.jpg"
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    # norm_img = (utils.normalize(img))

    print img.shape
    (hist, hog_img) = hog(img, orientations=36, pixels_per_cell=(2, 2),
                          cells_per_block=(2, 2), visualise=True, normalise=True)

    print hist
    print hist.shape
    # print_img = utils.denormalize(hog_img)
    # print print_img

    cv2.imshow('image', hog_img)
    cv2.waitKey(0)

main()
