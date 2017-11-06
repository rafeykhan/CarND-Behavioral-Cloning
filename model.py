import csv
import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image

CENTER, LEFT, RIGHT, STEERING, THROTTLE, BRAKE, SPEED = 0,1,2,3,4,5,6

def load_data(paths):
    data = []
    for path in paths:
        csv_log = path + 'driving_log.csv'
        with open(csv_log, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
    return data

def img_name(abs_path):
    path = abs_path.replace('\\','/').replace('IMG\r','IMG/ r').split('CarND-Behavioral-Cloning-P3/')[-1].split('/')
    path = '/'.join(p.strip() for p in path)
    return './'+path

def img_read(path):
    img = Image.open(path)
    return np.array(img)

def resize(img):
    """
    Resizes the images in the supplied tensor to the original dimensions of the NVIDIA model (66x200)
    """
    from keras.backend import tf as ktf
    return ktf.image.resize_images(img, [66, 200])

def generator(samples, batch_size, cameras= 1, flip= True):
    global CENTER, LEFT, RIGHT, STEERING

    while 1:
        shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            end= offset+batch_size
            batch_samples = samples[offset:end]

            images=[]
            angles=[]
            correction = 0.05

            for sample in batch_samples:
                c_img = img_read(img_name(sample[CENTER]))
                c_steer_angle = float(sample[STEERING])
                if cameras ==1:
                    images.append(c_img)
                    angles.append(c_steer_angle)
                    if flip:
                        c_img_flip = np.fliplr(c_img)
                        images.append(c_img_flip)
                        angles.append(-c_steer_angle)
                elif cameras ==3:
                    l_img = img_read(img_name(sample[LEFT]))
                    r_img = img_read(img_name(sample[RIGHT]))
                    images.extend([c_img, l_img, r_img])

                    l_steer_angle = c_steer_angle + 0.5*c_steer_angle +correction
                    r_steer_angle = c_steer_angle - 0.5*c_steer_angle -correction
                    angles.extend([c_steer_angle, l_steer_angle, r_steer_angle])

                    if flip:
                        c_img_flip = np.fliplr(c_img)
                        l_img_flip = np.fliplr(l_img)
                        r_img_flip = np.fliplr(r_img)
                        images.extend([c_img_flip, l_img_flip, r_img_flip])
                        angles.extend([-c_steer_angle, -l_steer_angle, -r_steer_angle])

            features = np.array(images)
            labels = np.array(angles)
            yield shuffle(features, labels)


from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout, Activation, ELU, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

def nvidia_model():
    #l2_lambda=0.0001
    #,W_regularizer=l2(l2_lambda)
    model = Sequential()
    
    #model.add(Cropping2D(cropping=((65,25),(0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x[:,65:-25,:,:], input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1.))
    model.add(Lambda(resize))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Flatten())
    
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Dense(1))
    model.summary()

    return model

def commaai_model():
    model = Sequential()
    model.add(Lambda(lambda x: x[:,65:-25,:,:], input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1.))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Flatten())
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Dense(1))
    model.summary()

    return model




## MAIN ##
a = './data-udacity/'
b = './data-selective/'
c = './data-reverse/'
d = './data-reverse-recover/'
e = './data-reverse-erratic/'
f = './data-reverse-selective/'
g = './data-reverse-2/'

paths = [c,d,e,f]

data = load_data(paths)
train_data, valid_data = train_test_split(data, test_size=0.2)

# compile and train the model using the generator function
batch_size=64
epochs = 5
cameras = 3
flip = True
rate = 0.001

train_generator = generator(train_data, batch_size=batch_size, cameras = cameras, flip = flip)
valid_generator = generator(valid_data, batch_size=batch_size, cameras = cameras, flip = flip)

model = nvidia_model()
from keras.models import load_model
#model = load_model('model.h5')
#model= commaai_model()


checkpoint = ModelCheckpoint('model.h5', monitor= 'val_loss', verbose=0, save_best_only=True, mode='auto')
model.compile(loss = 'mse', optimizer = Adam(lr = rate))

multiplier = cameras*(flip+1)
model.fit_generator(train_generator, samples_per_epoch=len(train_data)*multiplier, validation_data=valid_generator, nb_val_samples=len(valid_data)*multiplier, nb_epoch=epochs, callbacks=[checkpoint], verbose=1)


#model.save('model.h5')
print("Model saved.")
