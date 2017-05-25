import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import glob
from functions import *
import time
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    notcars = glob.glob(r"../non-vehicles/**/**/*.png")
    cars = glob.glob(r"../vehicles/**/**/*.png")

    color_spaces = ["RGB", "HSV", "LUV", "HLS", "YUV", "YCrCb"]
    orients = [3,6,9,12]  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off


    car_features = []
    notcar_features = []
    # calculate accuracy for different colorspace & orient combinates
    for color_space in color_spaces:
        for orient in orients:
            print('Using:',color_space,"color_space",orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
            car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
            notcar_features = extract_features(notcars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
            #pickle.dump({"car_features":car_features,"notcar_features":notcar_features}, open("features.p","wb"))

            X = np.vstack((car_features, notcar_features)).astype(np.float64)
            # Fit a per-column scalery_start_stop
            X_scaler = StandardScaler().fit(X)
            # Apply the scaler to X
            scaled_X = X_scaler.transform(X)

            # Define the labels vector
            y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
            X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

            print('Feature vector length:', len(X_train[0]))
            # Use a linear SVC
            svc = LinearSVC()
            # Check the training time for the SVC
            t=time.time()
            svc.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2-t, 2), 'Seconds to train SVC...')
            # Check the score of the SVC
            print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
            # Check the prediction time for a single sample
            t=time.time()
            # saving models under dir "models"
            pickle.dump({"network":svc,"X_scaler":X_scaler}, open("models/"+color_space+str(orient)+".p","wb"))
            car_features = []
            notcar_features = []
            print('---------'*6)

            """
            Accuracy results:
            Orient	                 Color Space
	                RGB	     HSV      LUV       HLS	     YUV	 YCrCb
            3	   0.9716   0.9828   0.9803	  0.9831	0.9885	0.9845
            6	   0.9837	0.9899	 0.9842	  0.9893	0.9901	0.993
            9	   0.9851	0.9907	 0.989	  0.9924	0.9924	0.9938
            12	   0.9845	0.9938	 0.987	  0.9904	0.993	0.9913
            """
