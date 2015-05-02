import math

import numpy as np
from scipy.optimize import minimize

import cv2


class BasisLights():
    W = np.array([1, 1, 1])
    SIGMA = 0.5
    EPSILON = 0.01

    def __init__(self, image_list, verbose=False):
        """
        """
        self.image_list = image_list
        self.verbose = verbose

    def fill(self, epsilon=0.01):
        """
        """

        # We get the size from the first image
        # Image shape is defined by (rows, cols, channels), see:
        # http://docs.opencv.org/trunk/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#accessing-image-properties
        img_shape = self.image_list[0].shape

        sum_weights_image = np.zeros_like(self.image_list[0])
        sum_image = np.zeros_like(self.image_list[0])
        print img_shape
        div_array = np.ones(img_shape)
        div_array = div_array * len(self.image_list)

        # Same shape as input images, but only 1 channel
        sum_weights = np.zeros((img_shape[0], img_shape[1], 1), np.float32)
        i_bar = np.zeros_like(sum_weights)

        for i, image in enumerate(self.image_list):
            weights = np.zeros_like(self.image_list[0])
            i_bar = np.dot(image, np.array([0.1140, 0.5870, 0.2990]))
            i_bar = np.reshape(i_bar, (i_bar.shape[0], i_bar.shape[1], 1))
            weights = i_bar/(i_bar + epsilon)
            if self.verbose:
                print "weights %s" % weights

            sum_weights += weights
            res = np.multiply(weights, image)
            sum_image += image
            sum_weights_image += res

        return sum_weights_image/sum_weights

    def edge(self):
        """
        """
        count = len(self.image_list)

        x0 = np.ones(count + 1)
        res = minimize(self.arg_min, x0, method='Nelder-Mead')

        for i in range(count):
            res[i] * self.image_list[i]

    def edge_weight(self):
        return 1

    def arg_min(self, l):
        rows = self.image_list[0].shape[0]
        cols = self.image_list[0].shape[1]

        gradient_map = self.gradient_map(self.image_list)

        for r in range(rows):
            for c in range(cols):
                # print "sum_lambda %s" % sum_lambda_int(l, r, c)
                # print "gradient map %s" % (gradient_map**2)
                # diff = np.gradient(sum_lambda_int(l, r, c))
                # print diff
                # gradient = np.arctan(diff[1]/diff[0]) * 180 / np.pi
                gradient = self.sum_lambda_int(l, r, c)
                res = self.edge_weight() * np.linalg.norm(
                    gradient - (gradient_map.item(r, c)**2))**2
        return res

    def orientation_map(self, mag, ori, threshold=0.01):
        return_image = np.zeros_like(mag)
        rows = mag.shape[0]
        cols = mag.shape[1]

        for r in range(rows):
            for c in range(cols):
                # for channel in range(channels):
                if mag.item(r, c) > threshold:
                    return_image.itemset(r, c,
                                         ori.item(r, c))
        return return_image

    def calculate_gradient_histograms(self, gradient_maps):
        return_map = np.zeros_like(gradient_maps[0])
        shape = gradient_maps[0].shape
        rows = shape[0]
        cols = shape[1]

        for r in range(rows):
            for c in range(cols):
                i = 0
                pixel_at_images = np.zeros_like(gradient_maps[0])
                for im in gradient_maps:
                    pixel_at_images.itemset(i, im.item(r, c))
                    i += 1
                hist = cv2.calcHist([pixel_at_images], [0], None, [36],
                                    [0, 180])

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
                return_map.itemset(r, c, cur_val)
        return return_map

    def gradient_map(self, images):
        gradient_maps = []
        for i in range(0, len(images)):

            # convert images to grayscale
            if len(images[0].shape) > 2:
                gray_image = cv2.cvtColor(images[i], cv2.cv.CV_BGR2GRAY)
            else:
                gray_image = images[i]

            sx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=-1)
            sy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=-1)

            # mag = cv2.magnitude(sx, sy)
            ori = cv2.phase(sx, sy, angleInDegrees=True)

            cv2.imshow('orientation', ori)
            cv2.waitKey(0)
            gradient_maps.append(ori)

        return self.calculate_gradient_histograms(gradient_maps)

    def diffuse_color(self):
        """
        """
        count = len(self.image_list)
        x0 = np.full(count, 1.0/count)
        lambdas = minimize(self.calc_diffuse_color, x0, method='Nelder-Mead')
        ret_image = self.sum_images(lambdas)

        return ret_image

    def angle(self, px1, px2):
        if (px1[0] == 0) and (px1[1] == 0) and (px1[2] == 0):
            t1 = np.zeros(3)
        else:
            t1 = px1 / np.linalg.norm(px1)

        if px2[0] == 0 and px2[1] == 0 and px2[2] == 0:
            t2 = np.zeros(3)
        else:
            t2 = px2 / np.linalg.norm(px2)

        dot = np.dot(t1, t2)
        if dot > 1:
            dot = 1
        elif dot < -1:
            dot = -1
        return math.acos(dot)

    def alpha(self, r, c):
        return math.exp(-(self.angle(self.px_avg(r, c), self.W)**2) /
                        (2*self.SIGMA**2))

    def px_avg(self, r, c):
        channels = self.image_list[0].shape[2]
        sum = np.zeros(channels)

        for img in self.image_list:
            px = img[r, c]
            sum += px

        avg = np.ones(channels)
        avg *= len(self.image_list)
        return sum / avg

    def sum_images(self, l):
        return_image = np.zeros_like(self.image_list[0])
        i = 0
        for img in self.image_list:
            return_image += np.multiply(l.item(i), img)
            i += 1

        return return_image

    def sum_lambda_int(self, l, r, c):
        sum = 0
        i = 0
        for img in self.image_list:
            px = img[r, c]
            sum = l[i] * px
            i += 1

        return sum

    def w_hat(self, l, r, c):
        return self.sum_lambda_int(l, r, c) / (self.sum_lambda_int(l, r, c) +
                                               self.EPSILON)

    def calc_diffuse_color(self, l):
        rows = self.image_list[0].shape[0]
        cols = self.image_list[0].shape[1]

        sum = 0
        for r in range(rows):
            for c in range(cols):
                sum_lambda = self.sum_lambda_int(l, r, c)
                sum += self.alpha(r, c) * self.angle(
                    sum_lambda, self.px_avg(r, c)) - (
                        1 - self.alpha(r, c)) * self.w_hat(
                            l, r, c) * self.angle(sum_lambda, self.W)

        print "lambda: %s" % l
        print "sum:    %s" % sum
        return sum

    # def fprime(self, x):
    #     ones = np.ones_like(x)
    #     y = np.subtract(x, ones)
    #     return dx
