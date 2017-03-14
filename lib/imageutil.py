import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageUtil:

    @staticmethod
    def image_console(main_screen, sc1, sc2):

        # assemble the screen example
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        if main_screen is not None: canvas[0:720, 0:1280] = main_screen

        if sc1 is not None: canvas[0:180, 0:320] = cv2.resize(sc1, (320, 180), interpolation=cv2.INTER_AREA)
        if sc2 is not None: canvas[0:180, 320:640] = cv2.resize(sc2, (320, 180), interpolation=cv2.INTER_AREA)

        return canvas

    @staticmethod
    def convert_2d_to_3d(heatmap):
        cmap = plt.get_cmap('hot')
        heatmap_img = cmap(heatmap)
        heatmap_img = np.delete(heatmap_img, 3, 2)
        return heatmap_img

    @staticmethod
    def load_cars_notcars_images():
        # Divide up into cars and notcars
        cars_path = 'images/vehicles/*/*.png'
        notcars_path = 'images/non-vehicles/*/*.png'
        cars = glob.glob(cars_path)
        notcars = glob.glob(notcars_path)
        return cars, notcars

    @staticmethod
    def convert_color(img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

    @staticmethod
    def bin_spatial(img, size=(32, 32)):
        color1 = cv2.resize(img[:,:,0], size).ravel()
        color2 = cv2.resize(img[:,:,1], size).ravel()
        color3 = cv2.resize(img[:,:,2], size).ravel()
        return np.hstack((color1, color2, color3))

    @staticmethod
    def color_hist(img, nbins=32):    #bins_range=(0, 256)
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    @staticmethod    
    def draw_boxes(img, bboxes, color=(0, 0, 256), thick=6):
        """ Define a function to draw bounding boxes
        """
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy   