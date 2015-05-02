#!/usr/bin/python2
import os

import basis_lights
import cv2
import image_utils as utils
import modifier_lights


def read_images(directory, count, normalize=True):
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
        img = cv2.imread(img_name)
        if img is None:
            print("Unable to read image: %s " % (img_name))
            exit(1)
        # Normalize images to range [0..1] so that we can more easily
        # do calculations on them
        if(normalize):
            img = utils.normalize(img)
        img_list.append(img)

    return img_list


def fill_light(directory, output, count=-1, verbose=False):
    """
    This function calculates the fill light of the images.

    Arguments:
    directory -- the directory where the images are located
    count     -- the number of imags to use for this calculation
    verbose   -- should we print debug info
    """

    img_list = read_images(directory, count)
    basis = basis_lights.BasisLights(img_list, verbose=verbose)
    res_image = basis.fill()
    cv2.imwrite(output, utils.denormalize(res_image))
    cv2.imshow('Fill light', res_image)
    cv2.waitKey(0)


def edge_light(directory, output, count=-1, verbose=False):
    """
    This function calculates the fill light of the images.

    Arguments:
    directory -- the directory where the images are located
    count     -- the number of imags to use for this calculation
    verbose   -- should we print debug info
    """
    pass


def diffuse_light(directory, output, count=-1, verbose=False):
    """
    This function calculates the fill light of the images.

    Arguments:
    directory -- the directory where the images are located
    count     -- the number of imags to use for this calculation
    verbose   -- should we print debug info
    """
    img_list = read_images(directory, count)
    basis = basis_lights.BasisLights(img_list, verbose=verbose)
    res_image = basis.diffuse_color()
    cv2.imwrite(output, utils.denormalize(res_image))
    cv2.imshow('Diffuse color light', res_image)
    cv2.waitKey(0)


def object_modifier(fill_light, edge_light, diffuse_light, output, mask,
                    fill=1, edge=1, diffuse=1, count=-1, verbose=False):
    """
    This function calculates the per object modifier for an image.

    Arguments:
    fill_light    -- path to the fill light image
    edge_light    -- path to the edge light image
    diffuse_light -- path to the diffuse color light image
    mask          -- the mask image that identifies the object of interest
    fill          -- the weight of fill light in the object
    edge          -- the weight of edge light in the object
    diffuse       -- the weight of diffuse light in the object
    count         -- the number of imags to use for this calculation
    verbose       -- should we print debug info
    """
    fill_image = utils.normalize(cv2.imread(fill_light))
    edge_image = utils.normalize(cv2.imread(edge_light))
    diffuse_image = utils.normalize(cv2.imread(diffuse_light))
    mask_image = utils.normalize(cv2.imread(mask))

    modifier = modifier_lights.ModifierLights(verbose=verbose)
    res_image = modifier.per_object(fill_image, edge_image, diffuse_image,
                                    mask_image, fill, edge, diffuse)
    cv2.imwrite(output, utils.denormalize(res_image))
    cv2.imshow('Object modifier', res_image)
    cv2.waitKey(0)


def soft_modifier(directory, output, sigma, count=-1, verbose=False):
    """
    This function calculates the soft lightning modifier for an image.

    Arguments:
    directory -- the directory where the images are located
    sigma     -- determines how far the lights get diffused
    count     -- the number of imags to use for this calculation
    verbose   -- should we print debug info
    """
    img_list = read_images(directory, count, normalize=False)
    modifier = modifier_lights.ModifierLights(img_list, verbose=verbose)
    res_image = modifier.soft(sigma)
    cv2.imwrite(output, utils.denormalize(res_image))
    cv2.imshow('Soft modifier', res_image)
    cv2.waitKey(0)


def regional_modifier(image_path, output, beta, verbose=False):
    """
    This function calculates the soft lightning modifier for an image.

    Arguments:
    image   -- the image to apply the modifier to
    beta    -- determines which areas will be emphasized
    count   -- the number of imags to use for this calculation
    verbose -- should we print debug info
    """
    image = cv2.imread(image_path)
    modifier = modifier_lights.ModifierLights(verbose=verbose)
    res_image = modifier.regional(image, beta)
    cv2.imwrite(output, res_image)
    cv2.imshow('Regional modifier', res_image)
    cv2.waitKey(0)
