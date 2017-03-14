import cv2
import numpy as np
from scipy.ndimage.measurements import label

from hogfeatures import HogFeatureUtil
from imageutil import ImageUtil
from heatmap import HeatmapUtil

class VehicleDetector:

    @staticmethod        
    def filter_false_positives(boxes):
        # TODO: Real-world scenario requires more sophisticated rules for 
        # handling objects as a false positives
        min_tolerance = 48
        max_tolerance = 150
        box_list = []
        for bbox in boxes:
            if abs(bbox[1][0] - bbox[0][0]) >= min_tolerance and abs(bbox[1][1] - bbox[0][1]) <= max_tolerance:
                box_list.append(bbox)
            else:
                print('Skip... object is too big or too small')
        return box_list
        
    @staticmethod
    def find_heat(img, box_list, threshold=1):
        # Add heat to each box in box list
        heat = np.zeros_like(img[:,:,0]).astype(np.float) 
        heat = HeatmapUtil.add_heat(heat,box_list)
            
        # Apply threshold to help remove false positives
        heat = HeatmapUtil.apply_threshold(heat, threshold)

        return heat
        
    @staticmethod
    def get_heatmap_labels(heat, threshold=1):

        heat = HeatmapUtil.apply_threshold(heat, threshold)
        
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)        
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        bboxes = HeatmapUtil.get_labeled_bboxes(labels)

        return bboxes, heatmap
    
    @staticmethod
    def get_windows(img_shape):
        """
        Get windows with different scales
        :param img_shape:
        :return:
        """
        windows = []
        windows_slides = [((64, 64),  [400, 500]),
           ((96, 96),  [400, 500]),
           ((128, 128),[450, 578]),
           ((192, 192),[450, 720])]

        for xy_window, y_start_stop in windows_slides:
            windodws_segment = VehicleDetector.__slide_window(img_shape, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                                 xy_window=xy_window, xy_overlap=(0.5, 0.5))
            windows = windows + windodws_segment
        return windows
    
    @staticmethod    
    def get_hot_windows(img, windows, svc, X_scaler, color_space = 'YCrCb', orient = 9,
                            pix_per_cell = 8, cell_per_block = 2, hog_channel = "ALL",
                            spatial_size = (32, 32), hist_bins = 32, spatial_feat = False,
                            hist_feat = True, hog_feat = True):
        """
        Get windows where the vehicles are detected
        :param img:
        :param windows:
        :param svc:
        :param X_scaler:
        :param color_space:
        :param orient:
        :param pix_per_cell:
        :param cell_per_block:
        :param hog_channel:
        :param spatial_size:
        :param hist_bins:
        :param spatial_feat:
        :param hist_feat:
        :param hog_feat:
        :return:
        """

        hot_windows = VehicleDetector.__search_windows(img, windows, svc, X_scaler, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)                       

        return hot_windows
    
    # Define a function you will pass an image 
    # and the list of windows to be searched (output of slide_windows())
    @staticmethod
    def __search_windows(img, windows, clf, scaler, color_space='YCrCb', 
                        spatial_size=(32, 32), hist_bins=32, 
                        hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, 
                        hog_channel="ALL", spatial_feat=False, 
                        hist_feat=True, hog_feat=True):
    
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            #4) Extract features for that window using single_img_features()
            features = HogFeatureUtil.single_img_features(test_img, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
            #5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = clf.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows

    @staticmethod
    def __slide_window(img_shape, x_start_stop=[None, None], y_start_stop=[None, None],
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img_shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img_shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list
