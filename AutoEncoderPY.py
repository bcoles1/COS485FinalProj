# Import all the required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import keras
from keras.datasets import mnist
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import imageio
import cv2 as cv
import glob
from pathlib import Path
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf

'''
Here we process them based on the desired input into the network.

1 - We just input the gray scale difference of each images and the next one.

2 - We input (1) and the original colour image.

3 - We input (2) and the gray scale difference of each images and its next to images.

'''
def DataProcessing(desiredInput):
    inputt = []
    Input2 = []
    masks = []
    overpassImages = []
    overpassMasks = []
    highwayImages = []
    highwayMasks = []   
    sofaImages = []
    sofaMasks = []

    #dataset2014/dataset/dynamicBackground/overpass/input
    files = glob.glob('TESTIMAGESET/*.jpg')
    for img in files:
        overpassImages.append(img)
    
    #dataset2014/dataset/baseline/highway/input
    files = glob.glob('TESTIMAGESET/*.jpg')   
    for img in files:
        highwayImages.append(img)
    
    #dataset2014/dataset/intermittentObjectMotion/sofa/input
    files = glob.glob('TESTIMAGESET/*.jpg')
    for img in files:
        sofaImages.append(img)
    
    #dataset2014/dataset/dynamicBackground/overpass/groundtruth
    files = glob.glob('TESTIMAGESETMASKS/*.png')
    for mask in files:
        overpassMasks.append(mask)
    
    #dataset2014/dataset/baseline/highway/groundtruth
    files = glob.glob('TESTIMAGESETMASKS/*.png')
    for mask in files:
        highwayMasks.append(mask)
    
    #dataset2014/dataset/intermittentObjectMotion/sofa/groundtruth
    files = glob.glob('TESTIMAGESETMASKS/*.png')
    for mask in files:
        sofaMasks.append(mask)

    for i in range(len(overpassImages) - 1):
    #The 0 argument in both of these causes us to input the greyscale difference.
        image = cv.imread(overpassImages[i], 0)
        nextImage = cv.imread(overpassImages[i+1], 0)
        greyscaleDiff = np.subtract(nextImage, image)
        inputt.append(greyscaleDiff)
        mask = imageio.imread(overpassMasks[i])
        masks.append(mask)
        if (i != len(overpassImages) - 2 and desiredInput == 3):
            nextNextImage = cv.imread(overpassImages[i+2], 0)
            #Subtract from original or 
            greyscaleDiff = np.subtract(nextNextImage, image)
            Input2.append(greyscaleDiff)

    if(desiredInput == 3):
        inputt = inputt[:-1]
    
    for i in range(len(highwayImages) - 1):
        #The 0 argument in both of these causes us to input the greyscale difference.
        image = cv.imread(highwayImages[i], 0)
        nextImage = cv.imread(highwayImages[i+1], 0)
        greyscaleDiff = np.subtract(nextImage, image)
        inputt.append(greyscaleDiff)
        mask = imageio.imread(highwayMasks[i])
        masks.append(mask)
        if (i != len(highwayImages) - 2 and desiredInput == 3):
            nextNextImage = cv.imread(highwayImages[i+2], 0)
            #Subtract from original or 
            greyscaleDiff = np.subtract(nextNextImage, image)
            Input2.append(greyscaleDiff)

    if(desiredInput == 3):
        inputt = inputt[:-1]
    
    for i in range(len(sofaImages) - 1):
        #The 0 argument in both of these causes us to input the greyscale difference.
        image = cv.imread(sofaImages[i], 0)
        nextImage = cv.imread(sofaImages[i+1], 0)
        greyscaleDiff = np.subtract(nextImage, image)
        inputt.append(greyscaleDiff)
        mask = imageio.imread(sofaMasks[i])
        masks.append(mask)
        if (i != len(sofaImages) - 2 and desiredInput == 3):
            nextNextImage = cv.imread(sofaImages[i+2], 0)
            #Subtract from original or 
            greyscaleDiff = np.subtract(nextNextImage, image)
            Input2.append(greyscaleDiff)

    if(desiredInput == 3):
        inputt = inputt[:-1]

    masks = np.array(masks).reshape(-1, 240, 320, 1)/255
    inputt = np.array(inputt).reshape(-1, 240, 320, 1)/255

    if(desiredInput == 1):
        overpassImages = None
        highwayImages = None
        sofaImages = None
        return inputt, masks
    
    colourImages = []

    if (desiredInput == 2):
        reduc = 1
    else:
        reduc = 2
    
    for i in range(len(overpassImages) - reduc):
        colourImages.append(cv.imread(overpassImages[i], 1))
    overpassImages = None
    
    for i in range(len(highwayImages) - reduc):
        colourImages.append(cv.imread(highwayImages[i], 1))
    highwayImages = None

    for i in range(len(sofaImages) - reduc):
        colourImages.append(cv.imread(sofaImages[i], 1))
    sofaImages = None

    colourImages = np.array(colourImages).reshape(-1, 240, 320, 3)/255

    if (desiredInput == 2):
        inputt = np.concatenate((inputt, colourImages), axis = 3)
        colourImages = None
        return inputt, masks

    Input2 = np.array(Input2).reshape(-1, 240, 320, 1)/255
    inputt = np.concatenate((inputt, colourImages, Input2), axis = 3)
    Input2 = None
    colourImages = None
    return inputt, masks

#Need to lok at what were doing for concat - probably just comes down to making sure we nail the axis
#We've cut the number of feature in half compared to regular uNet and cut the number of epochs down to 5. 
#This way, we can actually run this thing on my poor computer. 

inputt, masks = DataProcessing(1)

def unet(input_size =(240, 320, 1)):
    inputImg = Input(input_size)
    
    #conv_in block -
    conv_in = Conv2D(64, 3, padding = 'same', activation = 'relu', )(inputImg)
    conv_in = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(conv_in)

    #downConvBlock1
    dcB1 = MaxPooling2D(pool_size=(2, 2))(conv_in)
    dcB1 = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(dcB1)
    dcB1 = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(dcB1)
    
    #downConvBlock2
    dcB2 = MaxPooling2D(pool_size=(2, 2))(dcB1)
    dcB2 = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(dcB2)
    dcB2 = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(dcB2)
    
    #downConvBlock3
    dcB3 = MaxPooling2D(pool_size=(2, 2))(dcB2)
    dcB3 = Conv2D(512, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(dcB3)
    dcB3 = Conv2D(512, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(dcB3)
    
    #downConvBlock4
    dcB4 = MaxPooling2D(pool_size=(2, 2))(dcB3)
    dcB4 = Conv2D(1024, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(dcB4)
    dcB4 = Conv2D(1024, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(dcB4)
    #MAKE A DECISION ON DROUPOUT HERE
    
    #upConvBlock1
    ucB1 = Conv2D(512, 2, padding = 'same',activation = 'relu', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(dcB4))
    ucB1 = concatenate([dcB3,ucB1], axis = 3)
    ucB1 = Conv2D(512, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ucB1)
    ucB1 = Conv2D(512, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ucB1)
    
    #upConvBlock2
    ucB2 = Conv2D(256, 2, padding = 'same',activation = 'relu', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(ucB1))
    ucB2 = concatenate([dcB2,ucB2], axis = 3)
    ucB2 = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ucB2)
    ucB2 = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ucB2)
    
    #upConvBlock3
    ucB3 = Conv2D(128, 2, padding = 'same',activation = 'relu', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(ucB2))
    ucB3 = concatenate([dcB1,ucB3], axis = 3)
    ucB3 = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ucB3)
    ucB3 = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ucB3)
    
    #upConvBlock4
    ucB4 = Conv2D(64, 2, padding = 'same',activation = 'relu', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(ucB3))
    ucB4 = concatenate([conv_in,ucB4], axis = 3)
    ucB4 = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ucB4)
    ucB4 = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ucB4)
    
    #conv_out block
    co = Conv2D(1, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ucB4)
    outputImg = Conv2D(1, 1, activation = 'sigmoid')(co)
    
    model = Model(input = inputImg, output = outputImg)
    model.compile(optimizer = keras.optimizers.Adam(lr = 1e-4), metrics = ['accuracy'], loss = 'mse')
    return model

model = unet()
model.summary()
model.fit(inputt, masks,batch_size=2, epochs=10)
model_json = model.to_json()
with open("modelV1Final_testtex.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("modelV1Final_testtex.h5")
print("Saved model")







