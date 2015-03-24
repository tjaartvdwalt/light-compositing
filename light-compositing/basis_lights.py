import numpy as np


class BasisLights():
    def __init__(self, image_list):
        """
        """
        self.image_list = image_list

    def fill(self, epsilon=0.01):
        """
        """
        
        # We get the size from the first image
        # Image shape gets rows, cols and channels:
        # http://docs.opencv.org/trunk/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#accessing-image-properties
        img_shape = self.image_list[0].shape

        return_array = np.ndarray(shape=img_shape, dtype=np.float32,
                                  buffer=np.zeros(img_shape, np.float32))
        print img_shape
        div_array = np.ones(img_shape)
        div_array = div_array * len(self.image_list)
        # div_array = np.ndarray(shape=img_shape, dtype=np.float32,
        #                           buffer=)

        sum_weights = np.zeros(img_shape)
        for i, image in enumerate(self.image_list):
            weights = np.zeros(img_shape)
            weights = image/(image + epsilon)
            sum_weights += weights
            # print("image")
            # print(image)
            # print("image + e")
            # print(image + epsilon)
            # print("weights")
            # print(weights)

            return_array = return_array + (image * weights)

        return return_array/sum_weights

    def edge(self):
        """
        """
        pass

    def diffuse_color(self):
        """
        """
        pass
