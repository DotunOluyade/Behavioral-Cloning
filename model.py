import csv
import cv2
import numpy as np
import random
import shutil
from keras.models import Sequential
from keras.layers import Flatten, Dense
from scipy import ndimage
from keras.layers.convolutional import Cropping2D,Conv2D,Convolution2D
from keras.layers.core import Dense,Dropout,Activation,Flatten,Lambda
from keras.layers.core import K
from keras.layers.core import Reshape
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import Cropping2D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
import numpy as np
import os
import tensorflow as tf
import math
from keras.models import Model, load_model
from keras.optimizers import Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.utils import shuffle

# Path to root directory of dataset
data_path = 'DATA/'

# Data augmentation techniques
def brightness(image):
    bright_image = 0.5 + np.random.uniform()
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:,:,2] = image[:,:,2]*bright_image
    return cv2.cvtColor(image,cv2.COLOR_HSV2RGB)

def shadow_random(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    max_x = 255
    max_y = 55
    vertices = np.array([[max_x,max_y,random()*max_x,random()*max_y]], dtype = np.int32)
    image = region_of_interest(image,vertices)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def random_blur(image):
    size = 1+int(np.random.uniform()*7)
    if(size%2 == 0):
        size = size + 1
    image = cv2.GaussianBlur(image,(size,size),0)
    return image

def noise(image):
    return random_noise(x, mode='gaussian')

def image_rotate(image):
    # Rotate image randomly to simulate camera jitter
    rotate = random.uniform(-1, 1)
    M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), rotate, 1)
    img = cv2.warpAffine(img, M, (image.shape[1], image.shape[0]))
    return img

#Horizontal and vertical shifts
#Shift  images vertically by a random number to simulate the effect of driving up or down the slope.
def image_shift(img,trans_range=80):
    # Shift image and transform steering angle
    shift_x = trans_range*np.random.uniform()-trans_range/2
    shift_y = 40*np.random.uniform()-40/2
    M = np.float32([[1,0,shift_x],[0,1,shift_y]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    ang_adj = shift_x/trans_range*2*0.2
    return img, ang_adj    

#Shadow augmentation
def random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image    

# correct left-bias with data augmentation
def flip_image_steer_angle(image, measurement):
    image_flip = np.fliplr(image)
    measurement_flip = -measurement
    return image_flip, measurement_flip


def getSamplesFromDirectory(data_path, skipHeader=False):
    """
    Returns the lines from a driving log with base directory `data_path`.
    """
    samples = []
    with open(data_path + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            samples.append(line)
    return samples

# get all images and measurements under the root directory data_path
# get dataset without generators
def get_data_set(steering_adj=0.2):        
    images = []
    measurements = []
   
    
    directories = [x[0] for x in os.walk(data_path)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    for directory in dataDirectories:
        samples = getSamplesFromDirectory(directory)
        for line in samples:
            print('loading...',line)
            source_path = line[0]
            # read images in RGB formart using ndimage.imread
            img_center = np.asarray(ndimage.imread(directory + '/IMG/'  + source_path.split('/')[-1]))
            measurement = float(line[3])
            images.append(img_center)
            measurements.append(measurement)
           
            if measurement:
                f_image, f_measurement = flip_image_steer_angle(img_center,measurement)
                images.append(f_image)
                measurements.append(f_measurement)
            
            b_image = brightness(img_center)
            images.append(b_image)
            measurements.append(measurement)
            '''
            r_image = random_shadow(img_center)
            images.append(r_image)
            measurements.append(measurement)

            s_image, ang_shift = image_shift(img_center)
            images.append(r_image)
            measurements.append(ang_shift)


            b_image= random_blur(img_center)
            images.append(b_image)
            measurements.append(measurement)
            '''

            # Use left and right images to keep the car in then center of the track
            # by adding or substracting a steering adjustment value
            for lr_path in line[1:3]:
                lr_image = np.asarray(ndimage.imread(directory + '/IMG/' + lr_path.split('/')[-1]))
                lr_measurement = float(line[3])

                if lr_path == line[1]:
                    lr_measurement = lr_measurement + steering_adj
                if lr_path == line[2]:
                    lr_measurement = lr_measurement - steering_adj
                images.append(lr_image)
                measurements.append(lr_measurement)

    if len(images) != len(measurements) :
        print('Unbalanced data set');
        return

    print('converting dataset to np array....')
    images = np.array(images)
    measurements = np.array(measurements)
    return images, measurements

# use generators to fetch samples in batches
def generator(samples, batch_size=32):   
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for img, measurement in batch_samples:
                # preprocess each image by data augmentation
                
                images.append(img)
                measurements.append(measurement)

                #flip all images and steering angle
                
                if measurement:
                    f_image, f_measurement = flip_image_steer_angle(img,measurement)
                    images.append(f_image)
                    measurements.append(f_measurement)

                b_image = brightness(img)
                images.append(b_image)
                measurements.append(measurement)

                r_image = random_shadow(img)
                images.append(r_image)
                measurements.append(measurement)

                s_image, ang_shift = image_shift(img)
                images.append(r_image)
                measurements.append(ang_shift)


                b_image= random_blur(img)
                images.append(b_image)
                measurements.append(measurement)
                
               
                img = np.array(images)
                angles = np.array(measurements)
                
                yield sklearn.utils.shuffle(img, angles)


# The NVDIA CNN Architecture implementation
def nvidia_net():
    keep_prob = 0.5

    model = Sequential()
    
    # pre-processing   
    model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(160,320,3)))  # crop image
    model.add(Lambda(lambda x: K.tf.image.resize_images(x, (66,200)))) # resize image
    model.add(Lambda(lambda x: x/255.0 - 0.5)) # normalization
   
    # Convnet
    model.add(Conv2D(24,(5,5),activation="elu",strides=(2,2)))  
    model.add(Conv2D(36,(5,5),activation="elu",strides=(2,2)))
    model.add(Conv2D(48,(5,5),activation="elu",strides=(2,2)))
    model.add(Conv2D(64,(3,3),activation="elu"))
    model.add(Conv2D(64,(3,3),activation="elu"))
    model.add(Dropout(keep_prob))
   
    # FC
    model.add(Flatten())
    model.add(Dense(100,activation="elu"))
    model.add(Dense(50,activation="elu"))
    model.add(Dense(10,activation="elu"))
    model.add(Dense(1))

    return(model)
        

def train_network(do_Train):
    epoch = 5
    batch_size = 128
    validation_split = 0.2
    steering_adjustment = 0.28

    if do_Train == True:
        
        # fetch dataset
        images, measurements = get_data_set(steering_adj=steering_adjustment)
        
        '''
        # with generators approach
        print('Total Images: {}'.format( len(images)))
  
        samples = list(zip(images, measurements))
        train_samples, validation_samples = train_test_split(samples, test_size=validation_split)
        
        train_generator = generator(train_samples,batch_size)
        validation_generator = generator(validation_samples,batch_size)
        '''
        
        
    
        model = nvidia_net()
        model.compile(loss='mse', optimizer='adam')

        # saves the model weights after each epoch if the validation loss decreased
        checkpoint_callback = ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)
 
        history_object = model.fit(
            np.array(images), np.array(measurements), 
            nb_epoch=epoch, validation_split=validation_split, 
            shuffle=True, batch_size=batch_size)

        '''
        history_object = model.fit_generator(
                                        train_generator, steps_per_epoch=np.ceil(len(train_samples) // batch_size), 
                                        validation_data=validation_generator, 
                                        validation_steps=np.ceil(len(validation_samples) // batch_size), 
                                        epochs=epoch, verbose=1)      
        '''
        
        model.save('model.h5')
    else:
        model = load_model('model.h5')
    return history_object

do_Train = True
history_object = train_network(do_Train)

    
    