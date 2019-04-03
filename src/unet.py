"""                     
This script contains the implementation of the U-Net model, which is popular for
semantic segmentation of images.
    References:
    ----------
    1) https://lmb.informatik.uni-freiburg.de/people/ronneber/
    2) https://arxiv.org/abs/1505.04597 (U-Net: Convolutional Networks for Biomedical Image Segmentation

---------------------------
Author : Rutu Gandhi
"""


import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from src.data_loader import get_json_output, mask_to_region
from src.utils.data_prepare import unet_data_prepare
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as keras
from keras.layers.normalization import BatchNormalization as bn
from keras.preprocessing.image import *
import cv2


class UNet:
    def __init__(self, smooth=1, l2_lambda=0.0002, DropP=0.3, kernel_size=3, data ):
        """
        Initializes UNet class with the following parameters.

        Arguments
        ---------
        smooth : float
            Smoothing parameter to negate division by zero error
        l2_lambda : float
            Learning rate for the optimizer   
        DropP : float
            Dropping rate to prevent overfitting
        kernel_size : int
            Length of convolutional window size
	data : string
	    The name of folder that contains the train, mask and test
	    folders
        """
        self.smooth = smooth
        self.l2_lambda = l2_lambda
        self.DropP = DropP
        self.kernel_size = kernel_size
	self.data = data


    def dice_coef(self, y_true, y_pred):
        """ 
        The dice coefficient is a metric to calculate the similarilty
        (intersection) between the true values and the predictions.

        Arguments
        ---------
            y_true : float
                Ground truth
            y_pred : float
                Prediction
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f )
        return (2. * intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)


    def dice_coef_loss(self, y_true, y_pred):
        """
        Loss for dice-coefficient metric.

        Arguments
        ---------
            y_true : float
                Ground truth
            y_pred : float
                Prediction

        """
        return -dice_coef(y_true, y_pred)


    def unet(self, input_shape,learn_rate=1e-3):
        """
        Creates a U-Net model with it's corresponding convolutional layer network

        Arguments
        ---------
            input_shape : tuple
                Shape of the input images
            learn_rate : float
                Learning rate of the model

        Returns
        -------
            model : U-Net model
        """
        inputs = Input(input_shape)

        conv1 = Conv2D( 32, (self.kernel_size, self.kernel_size), activation='relu', 
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(inputs)
        conv1 = bn()(conv1)
        conv1 = Conv2D(32, (self.kernel_size, self.kernel_size), activation='relu', 
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda)  )(conv1)
        conv1 = bn()(conv1)


        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        pool1 = Dropout(self.DropP)(pool1)

        conv2 = Conv2D(64, (self.kernel_size, self.kernel_size), activation='relu', 
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(pool1)
        conv2 = bn()(conv2)

        conv2 = Conv2D(64, (self.kernel_size, self.kernel_size), activation='relu',
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(conv2)
        conv2 = bn()(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        pool2 = Dropout(self.DropP)(pool2)

        conv3 = Conv2D(128, (self.kernel_size, self.kernel_size), activation='relu', 
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(pool2)
        conv3 = bn()(conv3)
        conv3 = Conv2D(128, (self.kernel_size, self.kernel_size), activation='relu', 
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(conv3)
        conv3 = bn()(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = Dropout(self.DropP)(pool3)

        conv4 = Conv2D(256, (self.kernel_size, self.kernel_size), activation='relu',
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(pool3)
        conv4 = bn()(conv4)
        conv4 = Conv2D(256, (self.kernel_size, self.kernel_size), activation='relu',
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(conv4)
        conv4 = bn()(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        pool4 = Dropout(self.DropP)(pool4)

        conv5 = Conv2D(512, (self.kernel_size, self.kernel_size), activation='relu',
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(pool4)
        conv5 = bn()(conv5)
        conv5 = Conv2D(512, (self.kernel_size, self.kernel_size), activation='relu',
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(conv5)
        conv5 = bn()(conv5)

        up6 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2),
                          padding='same')(conv5), conv4],name='up6', axis=3)
        up6 = Dropout(self.DropP)(up6)


        conv6 = Conv2D(256, (self.kernel_size, self.kernel_size), activation='relu',
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(up6)
        conv6 = bn()(conv6)
        conv6 = Conv2D(256, (self.kernel_size, self.kernel_size), activation='relu',
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(conv6)
        conv6 = bn()(conv6)

        up7 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), 
            padding='same')(conv6), conv3],name='up7', axis=3)
        up7 = Dropout(self.DropP)(up7)

        conv7 = Conv2D(128, (self.kernel_size, self.kernel_size), activation='relu',
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(up7)
        conv7 = bn()(conv7)
        conv7 = Conv2D(128, (self.kernel_size, self.kernel_size), activation='relu',
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(conv7)
        conv7 = bn()(conv7)

        up8 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), 
            padding='same')(conv7), conv2],name='up8', axis=3)
        up8 = Dropout(self.DropP)(up8)

        conv8 = Conv2D(64, (self.kernel_size, self.kernel_size), activation='relu', 
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(up8)
        conv8 = bn()(conv8)
        conv8 = Conv2D(64, (self.kernel_size, self.kernel_size), activation='relu', 
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(conv8)
        conv8 = bn()(conv8)

        up9 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), 
            padding='same')(conv8), conv1],name='up9',axis=3)
        up9 = Dropout(self.DropP)(up9)

        conv9 = Conv2D(32, (self.kernel_size, self.kernel_size), activation='relu', 
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(up9)
        conv9 = bn()(conv9)
        conv9 = Conv2D(32, (self.kernel_size, self.kernel_size), activation='relu', 
            padding='same', kernel_regularizer=regularizers.l2(self.l2_lambda) )(conv9)
        conv9 = bn()(conv9)

        conv10 = Conv2D(3, (1, 1), activation='sigmoid', name='conv10')(conv9)

        model = Model(input = inputs, output = conv10)
        model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, 
            metrics = [dice_coef])
        model.summary()

        return model


    def train(self, input_shape, x_train_npy, y_train_npy):
        """
        Trains the U-Net model

        Arguments
        ---------
            input_shape : tuple
                Shape of input images
            x_train_npy : numpy file
                Numpy file for training images
            y_train_npy : numpy file
                Numpy file for training images
        """
        print("Loading data...")
        print("Loading data done!")
        model = unet(input_shape)
        print("U-Net has been trained.")
        es = EarlyStopping(monitor='loss', mode='min', verbose=1)
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', 
            verbose=1, save_best_only=True)
        
	x_train_npy, y_train_npy = unet_data_prepare()
	print('Fitting model on the test set...')
        model.fit(x_train_npy, y_train_npy, batch_size=1, epochs=10000, 
            verbose=1, shuffle=True, callbacks=[es])

    def predict(self, input_shape, x_test_npy, save_path):
	"""
	Predicts the masks for the test data
	Arguments
        ---------
            input_shape : tuple
                Shape of input images
            x_test_npy : numpy file
                Numpy file for test images
	    save_path : string
		Path for saving the masks
	"""
	if not os.path.exist(save_path):
            os.mkdir(save_path)
	nl=NeuronLoader()
	test_files = nl.test_files
	files = sorted(glob('~/neuron_dataset/test/*.test/images/image00000.tiff'))
	x_test=[]
	regions=[]
	for i in files:
	    img = cv2.imread(i)
	    x_test.append(img)
		
	x_test_npy = np.array(x_test)
	model = unet(input_shape)
	model.load_weights('unet.hdf5')
	model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, 
            metrics = [dice_coef])
	
        for i,im in enumerate(x_test_npy, start=0): 
            og_columns = im.shape[1]
            og_rows = im.shape[0]
            
            image = im[np.newaxis,...]
            image = image[...,np.newaxis]
            mask = model.predict(image)
            mask = mask[0,...]
            mask = mask[...,0]
            mask = cv2.resize(mask,(og_columns,og_rows))
		
            for x in range(0,mask.shape[0]):
                for y in range(0,mask.shape[1]):
                    if mask[x,y] >= 0.5:
                        mask[x,y] = 255
                    else:
                        mask[x,y] = 0
            cv2.imwrite(save_path + "/" + test_files[i] + ".tiff", mask)
	    region = mask_to_region(save_path + "/" + test_files[i] + ".tiff")
	    regions.append(region)
            print("................" + str(i) + ".......................")  
	datasets = [file[12:] for file in test_files]
	output = get_json_output(datasets, regions)
