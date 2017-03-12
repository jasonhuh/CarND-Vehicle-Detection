import cv2
import numpy as np
from scipy.ndimage.measurements import label

from hogfeatures import HogFeatureUtil
from imageutil import ImageUtil
from heatmap import HeatmapUtil

class VehicleDetector:
    
    @staticmethod
    def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient = 9, 
                            pix_per_cell = 8, cell_per_block = 2, spatial_size = (32, 32), hist_bins = 32):
    
        """ Define a single function that can extract features using hog sub-sampling and make predictions
        """
        print(X_scaler)
        
        box_list = []
        draw_img = np.copy(img)
#        img = img.astype(np.float32)/255
        
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = ImageUtil.convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1 
#        nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1 
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = HogFeatureUtil.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = HogFeatureUtil.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = HogFeatureUtil.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
    
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
    
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features
                spatial_features = ImageUtil.bin_spatial(subimg, size=spatial_size)
                hist_features = ImageUtil.color_hist(subimg, nbins=hist_bins)
    
                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    box = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart))
                    box_list.append(box)                
                    cv2.rectangle(draw_img,box[0],box[1],(0,0,255),6)
                    
        return box_list, draw_img    
        
    @staticmethod
    def apply_heat_map(img, box_list):
        # Add heat to each box in box list
        heat = np.zeros_like(img[:,:,0]).astype(np.float) 
        heat = HeatmapUtil.add_heat(heat,box_list)
            
        # Apply threshold to help remove false positives
        heat = HeatmapUtil.apply_threshold(heat,1)
        
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = HeatmapUtil.draw_labeled_bboxes(np.copy(img), labels)  
        return draw_img, heatmap
    
    @staticmethod    
    def search_and_classify(img, svc, X_scaler, color_space = 'YCrCb', y_start_stop = [400, 720], orient = 9, 
                            pix_per_cell = 8, cell_per_block = 2, hog_channel = "ALL",
                            spatial_size = (32, 32), hist_bins = 32, spatial_feat = True,
                            hist_feat = True, hog_feat = True):
        # Reduce the sample size because
        # The quiz evaluator times out after 13s of CPU time        
        
        windows = []
        window_sizes = [16, 32, 48, 64, 96, 128, 320]
        for ws in window_sizes:
            xy_window=(ws, ws)
            ans = VehicleDetector.__slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                             xy_window=xy_window, xy_overlap=(0.5, 0.5))
            windows = windows + ans
    
        hot_windows = VehicleDetector.__search_windows(img, windows, svc, X_scaler, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)                       

        
    #     # Add heat to each box in box list
    #     heat = add_heat(heat,hot_windows)
    
    #     # Apply threshold to help remove false positives
    #     heat = apply_threshold(heat,1)
    
    #     # Visualize the heatmap when displaying    
    #     heatmap = np.clip(heat, 0, 255)    
    #     labels = label(heatmap)
        
    #     draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    #     fig = plt.figure()
    #     plt.subplot(121)
    #     plt.imshow(draw_img)
    #     plt.title('Car Positions')
    #     plt.subplot(122)
    #     plt.imshow(heatmap, cmap='hot')
    #     plt.title('Heat Map')
    #     fig.tight_layout()
        return windows, hot_windows
    
    # Define a function you will pass an image 
    # and the list of windows to be searched (output of slide_windows())
    @staticmethod
    def __search_windows(img, windows, clf, scaler, color_space='YCrCb', 
                        spatial_size=(32, 32), hist_bins=32, 
                        hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, 
                        hog_channel="ALL", spatial_feat=True, 
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
    def __slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
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