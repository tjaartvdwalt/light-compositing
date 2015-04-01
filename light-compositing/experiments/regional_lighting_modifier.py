#!/usr/bin/python2
import numpy as np
import cv2
import sys

# from matplotlib.mlab import PCA
from sklearn.decomposition import PCA
sys.path.append("..")
import image_utils as utils


def convert_to_log(img):
    log_img = np.ones(img.shape, np.float32)
    with np.errstate(divide='ignore'):
        log_img = np.log(img, log_img)
        log_img[np.isneginf(log_img)] = 0

    return np.nan_to_num(log_img)


def calc_PCA(log_img):
    # Lets try matplotlib PCA
    # pca = PCA(np.transpose(log_img))

    # return np.transpose(pca.Y)
    
    # reshaped_img = log_img.reshape(1, -1)
    # print "reshaped %s %s " % (reshaped_img.shape)
    # pca = PCA(n_components=1, copy=True)
    # pca.fit(reshaped_img)
    # mean_reshaped = pca.transform(reshaped_img)
    # print "log_img %s %s " % log_img.shape
    # print(mean_reshaped.shape)
    # # mean_image = mean_reshaped.reshape(log_img.shape)
    # # print pca.components_[0].reshape(log_img.shape)
    #  # = pca.transform(log_img)
    # return mean_reshaped

    log_row = log_img.reshape(1, -1)
    mean, eigenvectors = cv2.PCACompute(log_row, maxComponents=1)
    print mean
    mean_image = mean.reshape(log_img.shape[0], -1)
    # cv2.imshow('image', mean_image)
    # cv2.waitKey(0)
    print mean_image
    return mean_image
    # print "mean %s" % mean
    # print "ev %s" % eigenvectors
    # return np.nan_to_num(eigenvectors)


def calc_PCA_hat(pca):
    pca_hat = np.zeros(pca.shape, np.float32)

    sum = 0
    count = pca.shape[0] * pca.shape[1]

    for r in range(pca.shape[0]):
        for c in range(pca.shape[1]):
            sum += pca.item(r, c)
            # print pca.item(r, c)
            # print sum

    avg = sum / count
    # print pca
    print "sum %s" % sum
    print "count %s " % count
    print "avg %s " % avg

    for r in range(pca.shape[0]):
        for c in range(pca.shape[1]):
            # print pca.item(r, c) - avg
            pca_hat.itemset(r, c, pca.item(r, c) - avg)

    return pca_hat


def user_map(pca_hat, beta=0):
    return_map = np.zeros(pca_hat.shape)

    for r in range(pca_hat.shape[0]):
        for c in range(pca_hat.shape[1]):
            # print "hat %f: " % (pca_hat.item(r, c))
            # print "exp %f: " % (beta * pca_hat.item(r, c))
            # print "val %f: " % np.exp(beta * pca_hat.item(r, c))
            return_map.itemset(r, c, np.exp(beta * pca_hat.item(r, c)))

    return return_map


def main():
    img_name = "../test_data/cafe/fill_light.png"
    img = cv2.imread(img_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    print img.dtype
    cv2.imshow('img', img)
    cv2.waitKey(0)
    # norm_img = utils.normalize(img)
    log_img = convert_to_log(img)
    # print log_img
    # for i in log_img:
    #     print i
    pca = calc_PCA(log_img)
    # cv2.imshow('pca', pca)
    # cv2.waitKey(0)
    # print pca
    # print pca.shape
    pca_hat = calc_PCA_hat(pca)
    # print pca_hat
    # cv2.imshow('pca_hat', pca_hat)
    # cv2.waitKey(0)

    my_map = user_map(pca_hat, beta=0.5)

    return_img = np.multiply(my_map, img)

    # cv2.imshow('my_map', my_map.astype("uint8"))
    # cv2.waitKey(0)
    return_img = np.clip(return_img, 0, 255)
    cv2.imshow('image', return_img.astype("uint8"))
    cv2.waitKey(0)
    # print return_img.astype("uint8")
main()
