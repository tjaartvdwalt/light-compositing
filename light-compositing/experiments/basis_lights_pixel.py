import numpy as np


def edge_lights(image_list, epsilon=0.01):
    """
    Produces an an image where each pixel in the resulting image is
    the average of the corresponding pixel in the input images.that is
    the average of all the input images.  All input images must be of
    the same size.

    @param image_list: A list of images to average
    @type image_list: list of cv2.image
    @return: A cv2.image of the same size as input images
    """

    # We get the size from the first image
    # Image shape gets rows, cols and channels:
    # http://docs.opencv.org/trunk/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#accessing-image-properties
    rows = image_list[0].shape[0]
    cols = image_list[0].shape[1]
    channels = image_list[0].shape[2]
    # dtype = image_list[0].dtype

    return_image = np.zeros((rows, cols, channels), np.float32)

    for r in range(rows):
        for c in range(cols):
            for channel in range(channels):
                sum_weighted_intensity = 0
                sum_weights = 0
                for image in image_list:
                    # print("%f" % image.item(r, c, channel))
                    intensity = image.item(r, c, channel)
                    # convert the intensity to range [0..1]
                    weight = intensity / (intensity + epsilon)
                    sum_weighted_intensity += weight * intensity
                    sum_weights += weight

                if(sum_weights == 0):
                    weighted_avg = 0
                else:
                    weighted_avg = sum_weighted_intensity / sum_weights * 1.0

                return_image.itemset((r, c, channel), weighted_avg)

    return return_image
