import scipy.ndimage as skimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import csv
import matplotlib.pyplot as plt
import random
from utils import *

# global variables 
USE_GRAY_IMAGES = True
PRE_STD_NORMALIZATION = True
CORRECTION_DELTA = 0.2
LEFT_IMAGES = []
RIGHT_IMAGES = []
CENTER_IMAGES = []
Y_CENTER = []
Y_LEFT = []
Y_RIGHT = []


# flipping image and steering measurement
is_gray_mode = lambda: USE_GRAY_IMAGES
flip_vertically = lambda image, angle: [np.fliplr(image), -1*angle]

#During training, you want to feed the left and 
#right camera images to your model as if they were 
#coming from the center camera. This way, you can teach 
#your model how to steer if the car drifts off to the 
#left or the right. Figuring out how much to add or 
#subtract from the center angle will involve some experimentation
def rgb2gray(img):
    # convert image from rgb to grayscale 
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    img = img.reshape(img.shape + (1,))
    return img

def create_random_shadow(img):
    # Simulate shadow due to sun on road
    # calculate the coordinates with respect 
    # to the formula y = mx + b
    # m = delta_y/delta_x; delta_x = distance(p1, p2) = p2 - p1
    # at (p2, 0) b = - m*x 
    height, width = img.shape[0], img.shape[1]
    
    # randomly choose p1 and p2
    [p1, p2] = np.random.permutation(np.arange(width))[:2]
    # calculate m 
    m = height / (p2 - p1)
    # calculate b with point of coordinate (p2, 0)
    b = - m*p1
    # change the pixel intensity at relevant coordinates
    for y in range (height):
        # calculate the corresponding x coordinate
        x = int((y - b) / m)
        # change the pixel intensity accordingly
        img[y, :x, :] = (img[y, :x, :] * .39).astype(np.int32)
    return img

def load_data (abs_filepath):
    with open(abs_filepath) as file:
        f_handle = csv.reader(file) 
        for line in f_handle:
            LEFT_IMAGES.append(line[0].strip())
            CENTER_IMAGES.append(line[1].strip())
            RIGHT_IMAGES.append(line[2].strip())

            angle = float(line[3])
            Y_CENTER.append(angle)
            Y_RIGHT.append(angle - CORRECTION_DELTA)
            Y_LEFT.append(angle + CORRECTION_DELTA)

def balance_data(datafiles, plot_data=False, reduce_factor=0.0):
    # datafiles: array containing csv fílepaths

    for datafile in datafiles:
        load_data(datafile)
    data = Y_LEFT + Y_CENTER + Y_RIGHT
    n, b, _ = plt.hist(data, bins=50, linewidth=0.2)
    if plot_data:
        plt.figure(1)
        plt.show()
    
    # retrieve the indices of bins
    indices_of_bins = np.digitize(data, b)

    # create a dataframe with all values
    df_left = pd.DataFrame({"images": LEFT_IMAGES, "orientation": ['left' for index in range(len(LEFT_IMAGES))], "angles": Y_LEFT})
    df_right = pd.DataFrame({"images": RIGHT_IMAGES, "orientation": ['right' for index in range(len(RIGHT_IMAGES))], "angles": Y_RIGHT})
    df_center = pd.DataFrame({"images": CENTER_IMAGES, "orientation": ['center' for index in range(len(CENTER_IMAGES))], "angles": Y_CENTER})

    _data = [df_left, df_center, df_right]
    df = pd.concat(_data)
    #:indices_of_bins}
    df["bindices"] = pd.Series(indices_of_bins)

    # drop unwanted peaks
    df = df.drop(df[(-0.23 < df['angles']) & (df['angles'] < -0.19)].sample(frac=0.6).index)
    df = df.drop(df[(0.19 < df['angles']) & (df['angles'] < 0.24)].sample(frac=0.6).index)
    df = df.drop(df[(-1. < df['angles']) & (df['angles'] < -0.74)].sample(frac=0.9).index)
    df = df.drop(df[(0.74 < df['angles']) & (df['angles'] < 1.0)].sample(frac=0.9).index)
    df = df.drop(df[(-1.25 < df['angles']) & (df['angles'] < -1.0)].sample(frac=1.0).index)
    df = df.drop(df[(1.0 < df['angles']) & (df['angles'] < 1.28)].sample(frac=1.0).index)

    # Reduce size of df, to check if the whole network works 
    if reduce_factor > 0.0:
        for indx in np.unique(df['bindices'].values):
            df = df.drop(df[df['bindices'] == indx].sample(frac=reduce_factor).index)
   
    plt.figure(num=2)
    _, _, _ = plt.hist(list(df["angles"]), bins=50, linewidth=0.2)

    if plot_data:
        plt.show()
    return df

def random_adjust_image(img):  
    # convert image to hsv
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 
    # split color channnels
    h, s, v = cv2.split(hsv_img)

    mul_factor = np.random.uniform(0.7, 1.4)
    np.multiply(v, mul_factor, out=v, casting='unsafe')
    mul_factor = np.random.uniform(0.7, 1.4)
    np.multiply(s, mul_factor, out=v, casting='unsafe')

    add_factor = np.random.uniform(-0.1, 0.1)
    np.add(v, add_factor, out=v, casting='unsafe')
    add_factor = np.random.uniform(-0.1, 0.1)
    np.add(s, add_factor, out=s, casting='unsafe')
    
    hsv_ = cv2.merge((h, s, v))
    result = cv2.cvtColor(hsv_, cv2.COLOR_HSV2RGB)
    return result


def normalize_pixel_values(img, gray=USE_GRAY_IMAGES, norm_std=PRE_STD_NORMALIZATION):
    # Zero center and normalize the pixel values
    if gray:
        img = rgb2gray(img)
        img -= np.mean(img)
        if norm_std:
            img /= np.std(img)
    else:
        img = img/255.0 - 1.0
    return img

def crop_and_resize(image, plot=False):
    cropped_img = image[70:130, :, :]
    resize_img = cv2.resize(cropped_img, (64, 16))
    if plot:
        plt.imshow(resize_img)
        plt.show()
    return resize_img

def pre_process_image(image):
    return normalize_pixel_values(crop_and_resize(image))

def load_all_data(samples):
    """
    samples: pf.Dataframe ['images', 'angles]
    imports data from the source folder and 
    returns X_train and y_train
    """
    images = []
    angles = []
    for index, row in samples.iterrows():
        images.append(pre_process_image(skimg.imread(row['images'])))
        angles.append(row['angles'])
    return np.array(images), np.array(angles)

@threadsafe_generator
def generator(samples, batch_size=256):
    """
    samples: pf.Dataframe ['images', 'angles]
    """
    num_samples = len(samples)

    while True:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = shuffle(samples[offset:offset+batch_size])
            images = []
            angles = []
            for _, row in batch_samples.iterrows():    
                # steering angle and associated image
                ang = row['angles']
                img = skimg.imread(row['images'])
                
                angles.append(ang)
                images.append(pre_process_image(img))
                # add flipped image if random value == 2
                rand_value = random.randint(1, 3)
                if rand_value == 1:
                    # associated image
                    if row['orientation'] == "left" or row['orientation'] == "right":
                        f_image, f_angle = flip_vertically(create_random_shadow(img), ang)
                        angles.append(f_angle)
                        images.append(pre_process_image(f_image))
                    else:
                        angles.append(ang)
                        images.append(pre_process_image(create_random_shadow(img)))    

                elif rand_value == 2:
                    angles.append(ang)
                    images.append(pre_process_image(random_adjust_image(img)))
                
                elif rand_value == 3:
                    if row['orientation'] == "left" or row['orientation'] == "right":
                        f_image, f_angle = flip_vertically(img, ang)
                        angles.append(f_angle)
                        images.append(pre_process_image(f_image))

            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)
