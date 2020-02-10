# -*- coding: utf-8 -*-
"""
@author: Powers
Architecture of neural network is inspired by the paper "U-Net: Convolutional 
Networks for Biomedical Image Segmentation", by Ronneberger, Fischer, and Brox.
"""


import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import concatenate
import time
import datetime




def make_model(rate):
    """
    Returns model which upscales 16x16 images to 64x64 images. 
    
    rate is learning rate for optimizer
    """
    
    inputs = keras.Input(shape=(16, 16, 3))
    
    #store inputs for skip connection before 16x16 upscale layers
    skip16 = inputs
   
    
    #16x16 downscale layers
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_16_down_0')(inputs)
    assert x.shape.as_list() == [None, 16, 16, 128]
    x = BatchNormalization(name='batch_norm_16_down_0')(x)
    x = LeakyReLU(name='leaky_re_lu_16_down_0')(x)
    
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_16_down_1')(x)
    assert x.shape.as_list() == [None, 16, 16, 128]
    x = BatchNormalization(name='batch_norm_16_down_1')(x)
    x = LeakyReLU(name='leaky_re_lu_16_down_1')(x)
    

    #MaxPool
    x = MaxPool2D(name='max_pool_8')(x)
    assert x.shape.as_list() == [None, 8, 8, 128]
    
    #store x for skip connection before 8x8 upscale layers
    skip8 = x
    
    
    #8x8 downscale conv layers
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_8_down_0')(x)
    assert x.shape.as_list() == [None, 8, 8, 128]
    x = BatchNormalization(name='batch_norm_8_down_0')(x)
    x = LeakyReLU(name='leaky_re_lu_8_down_0')(x)
    
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_8_down_1')(x)
    assert x.shape.as_list() == [None, 8, 8, 128]
    x = BatchNormalization(name='batch_norm_8_down_1')(x)
    x = LeakyReLU(name='leaky_re_lu_8_down_1')(x)
    
    
    #MaxPool
    x = MaxPool2D(name='max_pool_4')(x)
    assert x.shape.as_list() == [None, 4, 4, 128]
    
    #4x4 layers
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_4_0')(x)
    assert x.shape.as_list() == [None, 4, 4, 256]
    x = BatchNormalization(name='batch_norm_4_0')(x)
    x = LeakyReLU(name='leaky_re_lu_4_0')(x)
    
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_4_1')(x)
    assert x.shape.as_list() == [None, 4, 4, 256]
    x = BatchNormalization(name='batch_norm_4_1')(x)
    x = LeakyReLU(name='leaky_re_lu_4_1')(x)
    
    
    #Conv Transpose
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', 
                                     use_bias=False, name='conv2d_tr_8')(x)
    assert x.shape.as_list() == [None, 8, 8, 128]
    x = BatchNormalization(name='batch_norm_tr_8')(x)
    x = LeakyReLU(name='leaky_re_lu_tr_8')(x)

    #Skip connection
    x = concatenate([x, skip8])
    assert x.shape.as_list() == [None, 8, 8, 256]


    #8x8 upscale layers
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_8_up_0')(x)
    assert x.shape.as_list() == [None, 8, 8, 128]
    x = BatchNormalization(name='batch_norm_8_up_0')(x)
    x = LeakyReLU(name='leaky_re_lu_8_up_0')(x)
    
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_8_up_1')(x)
    assert x.shape.as_list() == [None, 8, 8, 128]
    x = BatchNormalization(name='batch_norm_8_up_1')(x)
    x = LeakyReLU(name='leaky_re_lu_8_up_1')(x)


    #Conv Transpose
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', 
                                     use_bias=False, name='conv2d_tr_16')(x)
    assert x.shape.as_list() == [None, 16, 16, 128]
    x = BatchNormalization(name='batch_norm_tr_16')(x)
    x = LeakyReLU(name='leaky_re_lu_tr_16')(x)

    #Skip connection
    x = concatenate([x, skip16])
    assert x.shape.as_list() == [None, 16, 16, 131]
    
    
    #16x16 upscale layers
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_16_up_0')(x)
    assert x.shape.as_list() == [None, 16, 16, 128]
    x = BatchNormalization(name='batch_norm_16_up_0')(x)
    x = LeakyReLU(name='leaky_re_lu_16_up_0')(x)
    
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_16_up_1')(x)
    assert x.shape.as_list() == [None, 16, 16, 128]
    x = BatchNormalization(name='batch_norm_16_up_1')(x)
    x = LeakyReLU(name='leaky_re_lu_16_up_1')(x)
    

    #Conv Transpose
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', 
                                     use_bias=False, name='conv2d_tr_32')(x)
    assert x.shape.as_list() == [None, 32, 32, 128]
    x = BatchNormalization(name='batch_norm_tr_32')(x)
    x = LeakyReLU(name='leaky_re_lu_tr_32')(x)
    
    
    #32x32 upscale layers
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_32_up_0')(x)
    assert x.shape.as_list() == [None, 32, 32, 128]
    x = BatchNormalization(name='batch_norm_32_up_0')(x)
    x = LeakyReLU(name='leaky_re_lu_32_up_0')(x)
    
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_32_up_1')(x)
    assert x.shape.as_list() == [None, 32, 32, 128]
    x = BatchNormalization(name='batch_norm_32_up_1')(x)
    x = LeakyReLU(name='leaky_re_lu_32_up_1')(x)
    
    
    #Conv Transpose
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', 
                                     use_bias=False, name='conv2d_tr_64')(x)
    assert x.shape.as_list() == [None, 64, 64, 64]
    x = BatchNormalization(name='batch_norm_tr_64')(x)
    x = LeakyReLU(name='leaky_re_lu_tr_64')(x)
    
    
    #64x64 upscale layers
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False,
               name='conv2d_64_up_0')(x)
    assert x.shape.as_list() == [None, 64, 64, 64]
    x = BatchNormalization(name='batch_norm_64_up_0')(x)
    x = LeakyReLU(name='leaky_re_lu_64_up_0')(x)
    
    outputs = Conv2D(3, (1, 1), strides=(1, 1), padding='same', activation='tanh', 
               name='conv2d_64_up_1')(x)
    assert outputs.shape.as_list() == [None, 64, 64, 3]
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    adam = tf.keras.optimizers.Adam(rate)
    model.compile( optimizer=adam, loss=keras.losses.MeanSquaredError() )
    
    return model



def load_images(indices, prefix, scale):
    """
    Returns a numpy array of images obtained by cropping those with given 
    indices. 
    
    prefix is leading part of file names
    scale is side length of images
    """
    images = np.zeros((indices.size, scale, scale, 3), dtype = np.float32)
    
    for k in range(indices.size):
        image_name = prefix + '_{:05d}.png'.format(indices[k])
        image = imageio.imread( image_name )
        images[k,:,:,:] = image
    
    #Normalize values to lie in [-1,1]
    images -= 127.5
    images /= 127.5

    return images

def createMiniBatches( X, batch_size ):
    """
    Returns a list of minibatches of size batchSize, constructed by randomly 
    partitioning data X.
    """
    
    np.random.shuffle(X)  #shuffle data
    
    numBatches = X.shape[0] // batch_size
    miniBatches = [None]*numBatches
    
    #partition X into minibatches
    for i in range(numBatches):   
        miniBatches[i] = X[ i*batch_size : (i+1)*batch_size ]
    
    if X.shape[0] % batch_size != 0:
        Xmini = X[ numBatches*batch_size : X.shape[0] ]
        miniBatches.append(Xmini)
    
    return miniBatches
  
    
def compute_val_loss(model):
    """
    Returns loss of model on validation set.
    """
    
    indices = np.arange(5000)
    minibatches = createMiniBatches(indices, 50)

    val_loss_sum = 0
    
    for i in range(len(minibatches)):   
        down_images = load_images( minibatches[i], 'images/val_images_downscaled2/val', 16)
        #Add fuzz to images
        down_fuzz_images = down_images + np.random.normal(scale=0.15, size=down_images.shape)
        #Clip values to lie in admissable range, [-1,1]
        down_fuzz_images = np.clip( down_fuzz_images, -1, 1 )
        real_images = load_images( minibatches[i], 'images/val_images/val', 64)
            
        val_loss_sum += model.test_on_batch(down_fuzz_images, real_images)
    
    return val_loss_sum/len(minibatches)


def upscale_and_save_images(model, epoch, tag):
    """
    Plots and saves 8 example images output of model, next to corresponding inputs.
    Input:
        model is the generator model
        epoch is the number of epochs through which model has been trained
        tag is a string used to determine where to store example images
    """
    
    indices = np.random.randint(5000, size = 8)
    down_images = load_images(indices, 'images/val_images_downscaled2/val', 16)
    #Add fuzz to images
    down_fuzz_images = down_images + np.random.normal(scale=0.15, size=down_images.shape)
    #Clip values to lie in admissable range, [-1,1]
    down_fuzz_images = np.clip( down_fuzz_images, -1, 1 )
    upscaled_images = model.predict(down_fuzz_images)

    plt.figure(figsize=(14,14))

    for i in range(8):
        plt.subplot(4, 4, 2*i+1)
        plt.imshow(down_fuzz_images[i, :, :, :] *0.5 + 0.5)
        plt.axis('off')
        plt.subplot(4, 4, 2*i+2)
        plt.imshow(upscaled_images[i, :, :, :] *0.5 + 0.5)
        plt.axis('off')

    plt.savefig('upscaled_images_' + tag + '/epoch_{:04d}.png'.format(epoch))
    plt.show()


def update_model( model, minibatch ):
    """
    Performs a single step of gradient descent on model. Returns loss 
    calculated on minbatch
    """
    down_images = load_images( minibatch, 'images/train_images_downscaled2/train', 16)
    #Add fuzz to images
    down_fuzz_images = down_images + np.random.normal(scale=0.15, size=down_images.shape)
    #Clip values to lie in admissable range, [-1,1]
    down_fuzz_images = np.clip( down_fuzz_images, -1, 1 )
    real_images = load_images( minibatch, 'images/train_images/train', 64)
    
    loss = model.train_on_batch( down_fuzz_images, real_images )
    
    return loss
    

def train_model( epochs, rate, prev_epochs, batch_size, tag ):
    """
    Returns trained model
    Input:
        epochs is number of epochs to train for
        rate is learning rate
        prev_epochs is number of epochs model has already been trained for
            (prev_epochs = 0 means that model is to be initialized for the 
            first time)
        batch_size is size of minibatches to be used for gradient descent
        tag is string indicating where to save example images and checkpoints
    """
    
    #Load previous losses. These aremoving averages of the Wasserstein loss at 
    #each gradient descent step. The averages are stored 4 times each epoch.
    train_losses = np.load('./losses_' + tag + '/train_losses_' + tag + '.npy')
    val_losses = np.load('./losses_' + tag + '/val_losses_' + tag + '.npy')
    
    model = make_model(rate)
    
    checkpoint_dir = './checkpoints_' + tag
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    num_images = len(os.listdir('./images/train_images'))
    
    indices = np.arange(num_images)
    
    for epoch in range(epochs):
        start = time.time()
        
        #Get moving average of test test losses so far:
        if prev_epochs == 0 and epoch == 0:
            loss_moving = 0
        else:
            loss_moving = train_losses[-1] 
        
        #We represent images by their indices to avoid having to hold all
        #images in memory:
        minibatches = createMiniBatches( indices, batch_size  )
        
        for i in range(len(minibatches)):               
            loss_mini = update_model( model, minibatches[i] )
            
            #Update loss moving average:
            loss_moving = 0.95 * loss_moving + 0.05 * loss_mini
            
            #Store loss 4 times per epoch (4th time will be at end of epoch):
            interval = len(minibatches) // 4
            if i == interval or i == 2*interval or i == 3*interval:
                train_losses = np.append(train_losses, loss_moving)
                
        # Save the model every epoch
        checkpoint.save(file_prefix = checkpoint_prefix)
        upscale_and_save_images(model, epoch + prev_epochs, tag)
        
        #Store train loss and validation loss
        train_losses = np.append(train_losses, loss_moving)
        np.save('./losses_' + tag + '/train_losses_' + tag + '.npy', train_losses)
        val_loss = compute_val_loss(model)
        val_losses = np.append(val_losses, val_loss)
        np.save('./losses_' + tag + '/val_losses_' + tag + '.npy', val_losses)
        
        print('Moving Average Loss: ' + str(loss_moving)) 
        print('Validation Loss: ' + str(val_loss))
        print('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
        current_time = datetime.datetime.now()
        print('Time: '+current_time.strftime("%I:%M:%S %p"))
   
    return model