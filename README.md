# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[img_carnd_vehicle_detection]: ./output_images/carnd_vehicle_detection.png
[img_cars_notcars]: ./output_images/cars_notcars.png
[img_hog_examples]: ./output_images/hog_examples.png
[img_training_output]: ./output_images/training_output.png
[img_heatmaps]: ./output_images/heatmaps.png
[img_detection_windows]: ./output_images/detection_windows.png
[img_video_snapshot]: ./output_images/video_snapshot.png

### Project Structure
The solution consists of one Jupyter Notebook and several custom python modules. Most of the model training, feature extraction and sliding window logics are split in the several python modules and stored in the "lib" folder, and I used the Jupyter Notebook as a client application that leverages these modules. Here is the overview of the project structure: 

![alt text][img_carnd_vehicle_detection]

- Jupyter Notebook file (CarND-Vehicle-Detection.ipynb): This is the entry point for the execution of the solution. This file is dependent on several custom modules as well as other python modules such as numpy and matplotlib.
- Trainer (lib/trainer.py): This class is responsible for training the detection of vehicles.
- HogFeaturesUtil (lib/hogfeatures.py): This class exposes several helper methods related to HOG features.
- VehicleDetector (lib/vehicle_detector.py): This class is responsible for detecting vehicles by leveraging the slide windows technique
- ImageUtil (lib/imageutil.py): This class exposes several useful image manipulation methods
- VideoUtil (lib/videoutil.py): This class exposes several useful video manipulation methods

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting hog features is contained in the `HogFeatureUtil.get_hog_features` method (`lib/hogfeatures.py`).

I prepared the training images in the following folder structure.
```
* vehicles:
    - images/vehicles/GTI_Far
    - images/vehicles/GTI_Left
    - images/vehicles/GTI_MiddleClose
    - images/vehicles/GTI_Right
    - images/vehicles/KITTI_extracted
* non-vehicles:
    - images/non-vehicles/Extras
    - images/non-vehicles/GTI
``` 

![alt text][img_cars_notcars]

As you can see in the 4th to 6th code cells in the Jupyter Notebook, to extract HOG features, I converted a training RGB image to a grayscaled image, and used the sciki-image hog() function with the following input parameters:

* orientations=9
* pixels_per_cell=8
* cells_per_block=2

![alt text][img_hog_examples]

#### 2. Explain how you settled on your final choice of HOG parameters.

The code for examining the HOG parameters is contained in the 7th code cell of Jupyter Notebook. In the cell, I leveraged `interact` attribute of ipywidgets library to explorer the different variations of the HOG parameters.

I tried many different variations of the HOG parameters and observed how the parameters impact the car detections. I ended up choosing this set of parameters:

* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* color_space = 'YCrCb'

​
Here are other combinations that also seemed to produce acceptable results:

* orient = 32
* pix_per_cell = 16
* cell_per_block = 2
* color_space = 'YCrCb'
​

In terms of color_space, 'YCrCb' seemed to be best choice even though 'LUV' also performed quite well. I chose 'YCrCb' instead of 'LUV' because LUV seemed to be too sensitive compared to 'YCrCb', resulting in too many false positives. On the flip side, 'YCrCb' was less sensitive, ending up missing some of vehicle objects.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used LinearSVC as the classifier for this project. The code for the training is contained in the Trainer class (`lib/trainer.py`). In the `Trainer.generate_train_and_test_data` method, the features were generated using the `HogFeatureUtil.extract_features` method. Then, the train data was scaled using StandardScaler for normalization. The features includes the HOG feature and histogram features.
 
Once the training data was generated from the above steps, the training data has been passed to the Trainer.train method where the training data was randomly shuffled, split to the train and test data, and then the train data has been fit using LinearSVC instance.

The train was complete with 98.73% of accuracy, and the model was saved as "model.pkl" for a future use.

![alt text][img_training_output]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for the sliding window search is contained in the `VehicleDetector.get_windows` method and `VehicleDetector.get_hot_windows` method (`lib/vehicle_detector.py`).

Considering that all objects including cars get smaller on a further distance, I created a grid of windows where the grids of windows are big (128x128) near a car dashboard and the grids of windows are smaller near (64x64) the horizon.

![alt text][img_detection_windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The code for the demonstration of how the pipeline works is contained in the 14th to 17th code cells of the Jupyter Notebook. Ultimately, I chose the sliding window based search with the aforementioned HOG parameters instead of the YCrCb 3-channel HOG features because, for some reason, the YCrCb 3-channel HOG features approach was significantly slower compared to the sliding window approach.

```
def visualize_find_cars(clf, X_scaler, original_img):

    img = original_img.astype(np.float32)/255   

    # Sliding window search
    windows = VehicleDetector.get_windows(img.shape)
    hot_windows = VehicleDetector.get_hot_windows(img, windows, clf, X_scaler, color_space=COLOR_SPACE,
                orient=ORIENT, pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK,
                hog_channel=HOG_CHANNEL, spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS)
    heat = VehicleDetector.find_heat(original_img, hot_windows, 1)
    heatmap_labels, heatmap = VehicleDetector.get_heatmap_labels(heat)

    # Draw result
    hot_windows_img = ImageUtil.draw_boxes(original_img, hot_windows)
    car_positions_img = ImageUtil.draw_boxes(original_img, heatmap_labels)     

    fig = plt.figure(figsize=(12, 36))
    plt.subplot(131)
    plt.imshow(hot_windows_img)
    plt.title('Detected boxes')
    plt.subplot(132)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    plt.subplot(133)
    plt.imshow(car_positions_img)
    plt.title('Car Positions')    
    fig.tight_layout()
```

Here are some test images:

![alt text][img_heatmaps]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a link to my video result:

[![link to my video result](https://img.youtube.com/vi/swTyZZCP-tM/0.jpg)](https://www.youtube.com/watch?v=swTyZZCP-tM)




#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For each frame, I captured the positions of the positive detections in a dequeue. Once the dequeue hold 10 positive detections, I generated the heatmap using the deque, and used the scipy.ndimage.measurements.label() method to generate the final positions.

### Here is a snapshot of a generated video:

![alt text][img_video_snapshot]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, I encountered the problems where the detected position of vehicles was jumping around for each frame with some false positives. I resolved this issue by storing the detected labels to a queue, and generating the final labels of vehicles by averaging the accumulated labels through heatmap and `label` function of `scipy.ndimage.measurements` library.

The pipeline may fail if the application encounters the following scenarios:

* A vehicle is not detected due to brightness, shadow or poor weather condition that either creates noise or hinders the system from detecting vehicles
* A strangely looking vehicle that does not look similar to train data.

I can further improve the performance and/or accuracy of the application by the following actions:

* Enhancing the training dataset to contain more diverse vehicle images and non-vehicle images. 
* Adjust the HOG parameters further to maximize the positive detections
* Leverage *Kalman filter* which is a much more robust algorithm that uses a series of measurements observed overtime.
* Instead of using the HOG features with SVC, use the YOLO real-time object detection
* Instead of using the HOG features with SVC, use the SSD multibox approach