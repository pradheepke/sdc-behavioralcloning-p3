import csv
import cv2
import numpy as np
import sklearn
from keras.callbacks import ModelCheckpoint

data_root_path = 'data/'
driving_log_file = 'data/driving_log.csv'
driving_log_from_sim_file = 'data_from_sim/trial1/driving_log.csv'
model_output_root = 'out/'
lines = []

def generator(samples, batch_size=16):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []

            for batch_sample in batch_samples:
                imgpath = data_root_path + batch_sample[0]
                # The data collected from simulator has absolute paths
                # If the path is absolute, no need to attach root
                if batch_sample[0][0] == '/':
                    imgpath = batch_sample[0]

                img = cv2.imread(imgpath)
                angle = float(batch_sample[3])
                images.append(img)
                angles.append(angle)
                
                # LR Flip data augmentation
                img_flipped = np.fliplr(img)
                angle_flipped = -angle
                images.append(img_flipped)
                angles.append(angle_flipped)
            
            X_train_batch = np.array(images)
            y_train_batch = np.array(angles)

            yield sklearn.utils.shuffle(X_train_batch, y_train_batch)
                    

def load_driving_log(driving_log_file):
    lines = []
    i = 0
    with open(driving_log_file) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            i = i + 1
            lines.append(line)
            #  For dev/debug
            #        if i >= 100:
            #            break
    return lines
    
lines = load_driving_log(driving_log_file)
lines.extend(load_driving_log(driving_log_from_sim_file))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(
   lines, test_size=0.2)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

# nVidia model architecture 
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# Model checkpoints are very useful to recover a model that was 
# better in an earlier epoch without needing to restart training
filepath="out/model-impr-{epoch:02d}-{val_mean_squared_error:.4f}.h5"

# For MSE we want to save the model everytime we find the best model
# so far according to validation accuracy. Best here is minimum.
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_mean_squared_error', 
                             save_best_only=True, mode='min')

model.fit_generator(train_generator, 
                    # times 2 because of augmentation from LR flip 
                    samples_per_epoch=len(train_samples) * 2,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples) * 2,
                    nb_epoch=10,
                    callbacks=[checkpoint])

model.save('out/model.h5')
