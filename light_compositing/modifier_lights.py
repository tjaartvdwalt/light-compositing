import numpy as np

import cv2


class ModifierLights():
    def __init__(self, image_list=None, verbose=False):
        """
        """
        self.image_list = image_list
        self.verbose = verbose

    def per_object(self, fill_image, edge_image, diffuse_image, mask_image,
                   fill_weight, edge_weight, diffuse_weight):
        """
        """
        fill_object = fill_image * mask_image
        edge_object = edge_image * mask_image
        diffuse_object = diffuse_image * mask_image

        masked_obj = (fill_object * fill_weight) / fill_weight
        + (edge_object * edge_weight) / edge_weight
        + (diffuse_object * diffuse_weight) / diffuse_weight

        inverse_mask = 1 - mask_image
        rest_image = inverse_mask * fill_image

        # alpha = 0.3
        # beta = 1.0 - alpha
        # comb_image = cv2.addWeighted(masked_obj, alpha, rest_image, beta, 0.0)

        cv2.imshow('Object modifier', masked_obj)
        cv2.waitKey(0)
        ret_image = masked_obj + rest_image
        # ret_image = cv2.bilateralFilter(comb_image, 5, 50, 50)

        return ret_image

    def soft(self, sigma):
        """
        """
        coef = self.get_mixture_coefficients(sigma)
        if self.verbose:
            print "Mixture coefficients: %s" % (coef)
        soft_light_img = self.sum_lambda_int(coef)
        hsv_image = cv2.cvtColor(soft_light_img, cv2.cv.CV_BGR2HSV)

        for i in range(hsv_image.shape[0]):
            hsv_image[i][:, 2] = hsv_image[i][:, 2] + 6

        return cv2.cvtColor(hsv_image, cv2.cv.CV_HSV2BGR)

    def get_mixture_coefficients(self, sigma):
        count = len(self.image_list)
        val = 1.0/count
        i_coef = np.transpose(np.full(count, val))
        s = self.S(sigma)
        s_times_lambda = np.dot(np.matrix(s), i_coef)
        n_coef = (np.linalg.norm(i_coef) /
                  np.linalg.norm(s_times_lambda)) * s_times_lambda

        return n_coef

    def S(self, sigma):
        count = len(self.image_list)
        soft_lambda = np.ones((count, count), dtype="float32")
        for i in range(count):
            for j in range(count):
                soft_lambda.itemset(i, j, self.element_s(
                    self.image_list[i], self.image_list[j], sigma))
        return soft_lambda

    def element_s(self, i, j, sigma):
        return np.exp(-(np.linalg.norm(i - j)**2)/(2 * (sigma**2)))

    def sum_lambda_int(self, l):
        # channels = img_list[0].shape[2]
        return_image = np.zeros_like(self.image_list[0])
        # sum = np.zeros(channels)
        i = 0
        for img in self.image_list:
            return_image += np.multiply(l.item(0, i), img)
            i += 1

        return return_image

    def regional(self, image, beta):
        """
        """
        hsv_image, intensity_image = self.get_hsv_image(image)

        log_img = self.convert_to_log(intensity_image)
        pca = self.calc_PCA(log_img)
        pca_hat = self.calc_PCA_hat(pca)
        filtered_pca_hat = cv2.bilateralFilter(pca_hat, 5, 50, 50)
        my_map = self.user_map(filtered_pca_hat, beta)
        new_intensity_img = np.multiply(my_map, intensity_image)
        new_intensity_img = np.clip(new_intensity_img, 0, 255)

        return_img = self.get_bgr_image(hsv_image,
                                        new_intensity_img.astype("uint8"))
        return return_img

    def get_hsv_image(self, img):
        hsv_image = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
        intensity_image = np.zeros(([hsv_image.shape[0], hsv_image.shape[1]]))
        for i in range(hsv_image.shape[0]):
            intensity_image[i] = hsv_image[i][:, 2]

        return hsv_image, intensity_image

    def get_bgr_image(self, hsv_image, new_intensity_image):
        for r in range(new_intensity_image.shape[0]):
            for c in range(new_intensity_image.shape[1]):
                hsv_image.itemset((r, c, 2), new_intensity_image.item(r, c))

        bgr_image = cv2.cvtColor(hsv_image, cv2.cv.CV_HSV2BGR)
        return bgr_image

    def convert_to_log(self, img):
        log_img = np.ones(img.shape, np.float32)
        with np.errstate(divide='ignore'):
            log_img = np.log(img, log_img)
            log_img[np.isneginf(log_img)] = 0

        return np.nan_to_num(log_img)

    def calc_PCA(self, log_img):
        log_row = log_img.reshape(1, -1)
        mean, eigenvectors = cv2.PCACompute(log_row, maxComponents=1)
        mean_image = mean.reshape(log_img.shape[0], -1)
        return mean_image

    def calc_PCA_hat(self, pca):
        pca_hat = np.zeros(pca.shape, np.float32)

        sum = 0
        count = pca.shape[0] * pca.shape[1]

        for r in range(pca.shape[0]):
            for c in range(pca.shape[1]):
                sum += pca.item(r, c)

        avg = sum / count

        for r in range(pca.shape[0]):
            for c in range(pca.shape[1]):
                pca_hat.itemset(r, c, pca.item(r, c) - avg)

        return pca_hat

    def user_map(self, pca_hat, beta=0):
        return_map = np.zeros(pca_hat.shape)

        for r in range(pca_hat.shape[0]):
            for c in range(pca_hat.shape[1]):
                return_map.itemset(r, c, np.exp(beta * pca_hat.item(r, c)))

        return return_map
