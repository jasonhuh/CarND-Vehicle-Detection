import time
import numpy as np

from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from hogfeatures import HogFeatureUtil

class Trainer:

    @staticmethod
    def generate_train_and_test_data(cars, notcars, sample_size=500, color_space='YCrCb', orient=32,
                     pix_per_cell=16, cell_per_block=2, hog_channel="ALL", spatial_size = (32, 32),
                                     hist_bins=32, spatial_feat = False, hist_feat = True, hog_feat = True):
        # Reduce the sample size because HOG features are slow to compute
        # The quiz evaluator times out after 13s of CPU time
        # cars = cars[0:sample_size]
        # notcars = notcars[0:sample_size]

        t = time.time()
        car_features = HogFeatureUtil.extract_features(cars, color_space=color_space,
                                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                                       orient=orient, pix_per_cell=pix_per_cell,
                                                       cell_per_block=cell_per_block,
                                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                       hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = HogFeatureUtil.extract_features(notcars, color_space=color_space,
                                                          spatial_size=spatial_size, hist_bins=hist_bins,
                                                          orient=orient, pix_per_cell=pix_per_cell,
                                                          cell_per_block=cell_per_block,
                                                          hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                          hist_feat=hist_feat, hog_feat=hog_feat)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        return X_scaler, scaled_X, y

    @staticmethod
    def train(svc, scaled_X, y, save_to='model.pkl'):
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
    
        print('Feature vector length:', len(X_train[0]))

        # Check the training time for the SVC
        t=time.time()
        
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

        Trainer.save_model(svc, save_to)
        return svc

    @staticmethod
    def save_model(svc, file_name='model.pkl'):
        # Save the trained model to a file system
        joblib.dump(svc, file_name)

    @staticmethod
    def load_model(file_name='model.pkl'):
        return joblib.load(file_name)