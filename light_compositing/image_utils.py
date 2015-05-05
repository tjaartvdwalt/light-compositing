import os

import numpy as np

import cv2


def read_image(image_path, normalize=True, downsample=0):
    image = cv2.imread(image_path)
    if image is None:
        print("Unable to read image: %s " % (image_path))
        exit(1)

    # Normalize images to range [0..1] so that we can more easily
    # do calculations on them
    if(normalize):
        image = normalize_img(image)

    return image


def read_images(directory, count, normalize=True, downsample=0, gray=False):
    """
    A helper function that reads the images in a given directory

    Arguments:
    directory -- the directory where the images are located
    count     -- the number of imags to use for this calculation

    Returns:
    A list of NumPy ndarrays containing the image data
    """
    # If we do not specify how many images to use, we will use all images
    if(count == -1):
        count = len(os.listdir(directory))

    img_list = []
    for i in range(0, count):
        img_name = "%s/%03d.png" % (directory, i)
        if(not gray):
            img = cv2.imread(img_name)
        else:
            img = cv2.imread(img_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if img is None:
            print("Unable to read image: %s " % (img_name))
            exit(1)
        # Normalize images to range [0..1] so that we can more easily
        # do calculations on them
        if(normalize):
            img = normalize_img(img)

        for i in range(0, downsample):
            img = cv2.pyrDown(img)

        img_list.append(img)

    return img_list


def normalize_img(img):
    div = np.ones(img.shape, np.float32)
    div = div / 255

    return img * div


def denormalize_img(img):
    clipped_img = np.clip(img, 0, 1)
    div = clipped_img * 255
    return np.uint8(div)
