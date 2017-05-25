import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from functions import *
from sklearn.cross_validation import train_test_split
import pickle
from PIL import Image
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# function which takes frame/image as input & return it after processing
def process_frame(img):
    out_img, heatmap = find_cars(img, ystart, ystop, scale, clf, scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    #show_two_img(out_img, heatmap, "Processed", "Heatmap", "test1result")
    heatmap = apply_threshold(heatmap,1)
    #show_single_img(heatmap, "Heatmap after threshold-2", "heatmap_after_threshold_2")
    labels = label(heatmap)
    #print(type(labels))
    out_img= draw_labeled_bboxes(np.copy(img), labels)
    return out_img

if __name__ == "__main__":
    # load parameters used while training
    dist_pickle = pickle.load( open("model.p", "rb" ) )
    clf = dist_pickle["network"]
    scaler = dist_pickle["X_scaler"]
    color_space = dist_pickle["color_space"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    hog_channel = dist_pickle["hog_channel"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    spatial_feat = dist_pickle["spatial_feat"]
    hist_feat = dist_pickle["hist_feat"]
    hog_feat = dist_pickle["hog_feat"]
    cars = dist_pickle["cars"]
    notcars = dist_pickle["notcars"]
    # search window parameters
    ystart = 350    # Min in y to search
    ystop = 656     # Max in y to search
    scale = 1.5

    #img = mpimg.imread('test_images/test1.jpg')
    #process_frame(img)
    #show_single_img(img, "result", "result")

    # process frames & generate video
    white_output = 'project_video_out.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_frame) # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    """
    # Visulaize Hog features
    car_img = cars[0]
    car_img_jpg = mpimg.imread(car_img)
    car_img = cv2.cvtColor(car_img_jpg, cv2.COLOR_RGB2YCrCb)
    notcar_img = notcars[10]
    notcar_img_jpg = mpimg.imread(notcar_img)
    notcar_img = cv2.cvtColor(notcar_img_jpg, cv2.COLOR_RGB2YCrCb)
    show_two_img(car_img_jpg, notcar_img_jpg, "Car RGB", "NotCar RGB", "car_not_car_rgb")
    show_two_img(car_img, notcar_img, "Car YCrCb", "NotCar YCrCb", "car_not_car_YCrCb")
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features,car_hog_image_ch0 = get_hog_features(car_img[:,:,0], orient, pix_per_cell, cell_per_block,vis=True, feature_vec=True)
    features,car_hog_image_ch1 = get_hog_features(car_img[:,:,1], orient, pix_per_cell, cell_per_block,vis=True, feature_vec=True)
    features,car_hog_image_ch2 = get_hog_features(car_img[:,:,2], orient, pix_per_cell, cell_per_block,vis=True, feature_vec=True)
    show_single_img(car_hog_image_ch0, "car hog features ch0", "car_hog_features_ch0")
    show_single_img(car_hog_image_ch1, "car hog features ch1", "car_hog_features_ch1")
    show_single_img(car_hog_image_ch2, "car hog features ch2", "car_hog_features_ch2")

    features,notcar_hog_image_ch0 = get_hog_features(notcar_img[:,:,0], orient, pix_per_cell, cell_per_block,vis=True, feature_vec=True)
    features,notcar_hog_image_ch1 = get_hog_features(notcar_img[:,:,1], orient, pix_per_cell, cell_per_block,vis=True, feature_vec=True)
    features,notcar_hog_image_ch2 = get_hog_features(notcar_img[:,:,2], orient, pix_per_cell, cell_per_block,vis=True, feature_vec=True)
    show_single_img(notcar_hog_image_ch0, "notcar hog features ch0", "notcar_hog_features_ch0")
    show_single_img(notcar_hog_image_ch1, "notcar hog features ch1", "notcar_hog_features_ch1")
    show_single_img(notcar_hog_image_ch2, "notcar hog features ch2", "notcar_hog_features_ch2")

    img = mpimg.imread('test_images/test5.jpg')
    draw_image = np.copy(img)
    img = img.astype(np.float32)/255
    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[ystart, ystop],
                        xy_window=(100, 100), xy_overlap=(0.5, 0.5))

    hot_windws = search_windows(img, windows, clf, scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        hist_range=(0, 256), orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
    window_img = draw_boxes(draw_image, hot_windws, color=(0, 0, 255), thick=6)
    show_single_img(window_img, "Test5 result", "test5result")
    """
