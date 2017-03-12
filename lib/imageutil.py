import glob
import cv2
import numpy as np

class ImageUtil:

    @staticmethod
    def image_console(mainScreen, sc1, sc2, sc3, sc4, text_arr):

        font = cv2.FONT_HERSHEY_COMPLEX
        textpanel = np.zeros((240, 640, 3), dtype=np.uint8)
        text_pos_y = 30
        for text in text_arr:
            cv2.putText(textpanel, text, (10, text_pos_y), font, 1, (255, 255, 255), 2)
            text_pos_y += 30

        # assemble the screen example
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        if mainScreen is not None: canvas[0:720, 0:1280] = mainScreen

        if sc2 is not None: canvas[0:240, 1280:1600] = cv2.resize(sc2, (320, 240), interpolation=cv2.INTER_AREA)
        if sc4 is not None: canvas[0:240, 1600:1920] = cv2.resize(sc4, (320, 240), interpolation=cv2.INTER_AREA)

        canvas[240:480, 1280:1920] = textpanel
        if sc3 is not None: canvas[720:1080, 0:1280] = cv2.resize(sc3, (1280, 360), interpolation=cv2.INTER_AREA) * 4

        return canvas

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
    def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
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