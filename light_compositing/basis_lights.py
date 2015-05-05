import math

import numpy as np
from scipy.optimize import minimize

import cv2


class BasisLights():
    W = np.array([1, 1, 1])
    SIGMA = 0.5
    EPSILON = 0.01

    def __init__(self, image_list, downsampled=None, verbose=False):
        """
        """
        self.image_list = image_list
        self.ds_image_list = downsampled
        self.verbose = verbose
        self.gmap = None
        self.weight = None

    def avg(self):
        """
        """
        sum = np.zeros_like(self.image_list[0])
        shape = self.image_list[0].shape
        count = np.full(shape, len(self.image_list))

        for i, image in enumerate(self.image_list):
            sum += image

        return sum / count

    def fill(self, epsilon=0.01):
        """
        """
        # We get the size from the first image
        # Image shape is defined by (rows, cols, channels), see:
        # http://docs.opencv.org/trunk/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#accessing-image-properties
        img_shape = self.image_list[0].shape

        sum_weights_image = np.zeros_like(self.image_list[0])
        sum_image = np.zeros_like(self.image_list[0])
        # div_array = np.ones(img_shape)
        # div_array = div_array * len(self.image_list)

        # Same shape as input images, but only 1 channel
        sum_weights = np.zeros((img_shape[0], img_shape[1], 1), np.float32)
        i_bar = np.zeros_like(sum_weights)

        for i, image in enumerate(self.image_list):
            my_weights = np.zeros_like(self.image_list[0])
            i_bar = np.dot(image, np.array([0.1140, 0.5870, 0.2990]))
            i_bar = np.reshape(i_bar, (i_bar.shape[0], i_bar.shape[1], 1))
            my_weights = i_bar/(i_bar + epsilon)
            print my_weights
            # if self.verbose:
            #     print "weights %s" % weights

            sum_weights += my_weights
            res = np.multiply(my_weights, image)
            sum_image += image
            sum_weights_image += res

        return sum_weights_image/sum_weights

    def edge(self):
        """
        """
        (self.gmap, self.weight) = self.gradient_map(self.ds_image_list)
        lambdas = self.optimize(self.calc_edge_light)
        ret_image = self.sum_images(lambdas, self.image_list)

        return ret_image

    def calc_edge_light(self, l):
        rows = self.ds_image_list[0].shape[0]
        cols = self.ds_image_list[0].shape[1]

        res = 0
        for r in range(rows):
            for c in range(cols):
                gradient = self.sum_lambda_int(l, r, c)
                res += self.weight.item(r, c) * np.linalg.norm(
                    gradient - (self.gmap.item(r, c)**2))**2

        if self.verbose:
            print "lambda: %s" % l
            print "sum:    %s" % res
        return res

    def optimize(self, my_function):
        bnds = []
        img_count = len(self.ds_image_list)
        for i in range(img_count):
            bnds.append((0, 1))

        x0 = np.full(img_count, 1.0/img_count)
        lambdas = minimize(my_function, x0, method='TNC', jac=False,
                           bounds=bnds)

        if self.verbose:
            print lambdas.message
            print "Final choice of lambdas = %s" % (lambdas.x)
        return lambdas.x

    def sum_images(self, l, images):
        return_image = np.zeros_like(images[0])

        for i, img in enumerate(images):
            return_image += np.multiply(l[i], img)

        return return_image

    def sum_lambda_int(self, l, r, c):
        sum = 0
        i = 0
        for img in self.ds_image_list:
            px = img[r, c]
            sum = l[i] * px
            i += 1

        return sum

    def orientation_map(self, mag, ori, threshold=0.01):
        return_image = np.zeros_like(mag)
        rows = mag.shape[0]
        cols = mag.shape[1]

        for r in range(rows):
            for c in range(cols):
                if mag.item(r, c) > threshold:
                    return_image.itemset(r, c,
                                         ori.item(r, c))
        return return_image

    def calcHist(self, oris, mags):
        bins = []
        for i in range(37):
            bins.append([])

        for i, ori in enumerate(oris):
            if(ori > 180):
                val = ori - 180
            else:
                val = ori
            index = int(val / 5)
            bins[index].append((ori, mags[i]))
        return bins

    def get_max_index(self, bins):
        max_value = 0
        max_index = 0

        for i, my_bin in enumerate(bins):
            if len(my_bin) > max_value:
                max_value = len(my_bin)
                max_index = i

        return max_index

    def get_max_weight(self, bins):
        max_index = self.get_max_index(bins)
        max_value = len(bins[max_index])

        total = 0
        for i, bin in enumerate(bins):
            total += len(bin)

        return 1.0 * max_value / total

    def get_max_mag_index(self, my_bin):
        max_value = 0
        max_index = 0
        for i, value in enumerate(my_bin):
            if value > max_value:
                max_value = value
                max_index = i

        return max_index

    def calc_gradient_map(self, gradient_maps):
        shape = gradient_maps[0][0].shape
        rows = shape[0]
        cols = shape[1]
        return_map = np.zeros((rows, cols))
        weight_map = np.zeros((rows, cols))

        for r in range(rows):
            for c in range(cols):
                ori_at_images = np.zeros((len(gradient_maps)),
                                         dtype=np.float32)
                mag_at_images = np.zeros((len(gradient_maps)),
                                         dtype=np.float32)
                for i, im in enumerate(gradient_maps[0]):
                    ori_at_images.itemset(i, im[r][c])

                for i, im in enumerate(gradient_maps[1]):
                    mag_at_images.itemset(i, im[r][c])

                hist = self.calcHist(ori_at_images, mag_at_images)
                index = self.get_max_index(hist)
                weight_map.itemset(r, c, self.get_max_weight(hist))
                max_mag_index = self.get_max_mag_index(hist[index])
                return_map.itemset(r, c, hist[index][max_mag_index][0])

        return (return_map, weight_map)

    def gradient_map(self, images):
        gmaps = []
        for i, image in enumerate(images):
            sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=-1)
            sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=-1)

            mag = cv2.magnitude(sx, sy)
            ori = cv2.phase(sx, sy, angleInDegrees=True)

            gmaps.append((ori, mag))

        (gmap, weight) = self.calc_gradient_map(gmaps)
        return (gmap, weight)

    def diffuse_color(self):
        """
        """
        lambdas = self.optimize(self.calc_diffuse_color)
        ret_image = self.sum_images(lambdas, self.image_list)

        return ret_image

    def calc_diffuse_color(self, l):
        rows = self.ds_image_list[0].shape[0]
        cols = self.ds_image_list[0].shape[1]

        my_sum = 0
        for r in range(rows):
            for c in range(cols):
                sum_lambda = self.sum_lambda_int(l, r, c)
                angle1 = self.angle(sum_lambda, self.px_avg(r, c))
                angle2 = self.angle(sum_lambda, self.W)
                alpha = self.alpha(r, c)
                inv_alpha = (1 - self.alpha(r, c))
                w_hat = self.w_hat(l, r, c)
                val = (alpha * angle1) - (inv_alpha * w_hat * angle2)
                my_sum += np.dot(val, np.array([0.1140, 0.5870, 0.2990]))
        if self.verbose:
            print "lambda: %s" % l
            print "sum:    %s" % my_sum
        return my_sum

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
        channels = self.ds_image_list[0].shape[2]
        sum = np.zeros(channels)

        for img in self.ds_image_list:
            px = img[r, c]
            sum += px

        avg = np.ones(channels)
        avg *= len(self.ds_image_list)
        return sum / avg

    def w_hat(self, l, r, c):
        sum_imgs = self.sum_lambda_int(l, r, c)
        return sum_imgs / (sum_imgs + self.EPSILON)
