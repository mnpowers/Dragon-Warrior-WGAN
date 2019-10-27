# -*- coding: utf-8 -*-
"""
@author: Powers
train_models function is based on tensorflow MNIST GAN tutorial
Wasserstein type details are from ``Wasserstein GAN'' paper by Arjovsky et al.
and ``Improved Training of Wasserstein GANs'' by Gulrajani et al.
"""

import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
import time
import datetime



def make_generator_model():
    """
    Returns model for generating images. 
    The model takes as input 64-dimensional vectors, and outputs 
    tensors of size 64x64x3, representing images.
    """
    model = tf.keras.Sequential()
    
    model.add(layers.Dense(128, use_bias=False, input_shape=(64,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(128, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(4*4*256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((4, 4, 256)))
    assert model.output_shape == (None, 4, 4, 256)
   
    model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model

def make_discriminator_model():
    """
    Returns model which assigns real values to images, in such a way that
    the average value on real images exceeds the average value on fake images
    by as large a margin as possible.
    Model takes as input 64x64x3 tensors, outputs a real number. 
    """
    
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    assert model.output_shape == (None, 32, 32, 256)
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same'))
    assert model.output_shape == (None, 32, 32, 256)
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    assert model.output_shape == (None, 4, 4, 512)
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(1))

    return model

def wasserstein_loss(real_output, fake_output):
    """
    Returns approximate (negative of the ) Wasserstein distance between the 
    distributions given by a list of real images and a list of fake images.
    Input to this function is the lists of outputs of discriminator on real
    images and fake images. We assume these two lists have the same length.
    """
    real_loss = tf.math.reduce_mean(real_output)
    fake_loss = tf.math.reduce_mean(fake_output)
    wass_loss = (fake_loss -real_loss)/2
    
    return wass_loss

def discriminator_loss(wass_loss, gradients, lambd):
    """
    Returns the full discriminator loss, defined to be the Wassertein loss
    with gradient penalty.
    Inputs:
        wass_loss is Wasserstein loss
        gradients is list of gradients for the discriminator
        lamd is weight given to gradient penalty
    """
    grad_penalties = ( tf.norm(gradients, axis = 1, keepdims=True) - 1 )**2
    grad_loss = tf.math.reduce_mean(grad_penalties)
    return wass_loss + lambd * grad_loss

def generator_loss(fake_output):
    """
    Returns the (negative of the) average output of discriminator on a 
    minibatch of fake images.
    """
    return -tf.math.reduce_mean(fake_output)


def crop_images(indices):
    """
    Returns a numpy array of images obtained by cropping those with given 
    indices.
    """
    cropped_images = np.zeros((indices.size, 64, 64, 3), dtype = np.float32)
    
    for k in range(indices.size):
        image_name = 'dw_images/dw_' + '{:05d}.png'.format(indices[k])
        image = imageio.imread( image_name )
        image_crop = image[87:151,104:168,:]
        cropped_images[k,:,:,:] = image_crop
    
    #Normalize values to lie in [-1,1]
    cropped_images -= 127.5
    cropped_images /= 127.5

    return cropped_images

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
  
def generate_and_save_images(model, epoch, noise_dim, tag):
    """
    Plots and saves 16 example images output by generator model.
    Input:
        model is the generator model
        epoch is the number of epochs through which model has been trained
        noise_dim is dimesion of input for model
        tag is a string used to determine where to store example images
    """
    
    noise = tf.random.normal([16, noise_dim])
    predictions = model(noise, training=False)

    plt.figure(figsize=(12,12))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :] *0.5 + 0.5)
        plt.axis('off')

    plt.savefig('generated_images_' + tag + '/epoch_{:06d}.png'.format(epoch))
    plt.show()

def sample_gen_and_wass_losses(gen_model, disc_model, num_images, noise_dim):
    """
    Returns generator loss and wasserstein loss for models, calculated using
    10*128 of real images and 10*128 fake images.
    Input:
        gen_model is generator model
        disc_model is discriminator model
        num_images is total number of real images in dataset
        noise_dim is dimension of input of gen_model
    """
    gen_tot = 0
    wass_tot = 0
    
    for i in range(10):
        noise = tf.random.normal([128, noise_dim])
        generated_images = gen_model(noise, training=False)
        fake_output = disc_model(generated_images, training=False)
        gen_tot += generator_loss(fake_output)  
        
        real_images = crop_images(np.random.randint(num_images, size = 128))
        real_output = disc_model(real_images, training=False)
        wass_tot += wasserstein_loss(real_output, fake_output)
    
    return gen_tot/10, wass_tot/10
    

def update_models( gen_model, disc_model, gen_optimizer, disc_optimizer,
                  real_images, noise, sample_size, lambd, train_gen ):
    """
    Performs a single step of gradient descent on disc_model, and also on 
    gen_model if train_gen == true
    """
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_tape.watch( gen_model.trainable_variables )
        disc_tape.watch( disc_model.trainable_variables )
        
        fake_images = gen_model(noise, training=train_gen)
        fake_output = disc_model(fake_images, training=True)
        real_output = disc_model(real_images, training=True)
        wass_loss = wasserstein_loss(real_output, fake_output)
        gen_loss = None
        if train_gen:
            gen_loss = generator_loss(fake_output)
        
        #For gradient penalty, we take randomly weighted averages
        #of real and fake images, and calculate gradients of
        #discriminator on these averages
        eps = tf.random.uniform((sample_size,1,1,1))  #Random weights
        av_images = eps * real_images + (1 - eps) * fake_images
        with tf.GradientTape() as gp_tape:
            gp_tape.watch( av_images )
            av_output = disc_model(av_images, training=True)
        av_im_grad = gp_tape.gradient(av_output, av_images)
        #convert list to tensor
        av_im_grad = tf.compat.v1.convert_to_tensor(av_im_grad)
        #reshape to matrix whose rows are gradients
        av_im_grad = tf.reshape(av_im_grad, (sample_size, 64*64*3))
        disc_loss = discriminator_loss(wass_loss, av_im_grad, lambd)
                
    disc_grad = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
    disc_optimizer.apply_gradients(zip(disc_grad, disc_model.trainable_variables))

    if train_gen:
        gen_grad = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grad, gen_model.trainable_variables))
    
    

def train_models( epochs, gen_rate, disc_rate, prev_epochs, batch_size, noise_dim, 
                           n_critic, tag, lambd = 10, num_images = 68237 ):
    """
    Returns trained generator and discriminator models
    Input:
        epochs is number of epochs to train for
        gen_rate is learning rate for generator model
        disc_rate is learning rate for discriminator model
        prev_epochs is number of epochs models have already trained for
            (prev_epochs = 0 means that models are to be initialized for the 
            first time)
        batch_size is size of minibatches to be used for gradient descent
        noise_dim is dimension of input of generator model
        n_critic is ratio of number of gradient descent steps for discriminator 
            to number of gradient descent steps for generator
        tag is string indicating where to save example images and checkpoints
        lambd is weight given to gradient penalty in discriminator loss
        num_images is number of real images in dataset   
    """
    
    gen_model = make_generator_model()
    disc_model = make_discriminator_model()
    
    gen_optimizer = tf.keras.optimizers.Adam(gen_rate, beta_1=0., beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(disc_rate, beta_1=0., beta_2=0.9)
    
    gen_checkpoint_dir = './generator_checkpoints_' + tag
    gen_checkpoint_prefix = os.path.join(gen_checkpoint_dir, "ckpt")
    gen_checkpoint = tf.train.Checkpoint(gen_optimizer=gen_optimizer,
                                 gen_model=gen_model)
    gen_checkpoint.restore(tf.train.latest_checkpoint(gen_checkpoint_dir))

    disc_checkpoint_dir = './discriminator_checkpoints_' + tag
    disc_checkpoint_prefix = os.path.join(disc_checkpoint_dir, "ckpt")
    disc_checkpoint = tf.train.Checkpoint(disc_optimizer=disc_optimizer,
                                 disc_model=disc_model)
    disc_checkpoint.restore(tf.train.latest_checkpoint(disc_checkpoint_dir))

    indices = np.arange(num_images)
    
    for epoch in range(epochs):
        start = time.time()
        #We represent images by their indices to avoid having to hold all
        #images in memory
        minibatches = createMiniBatches( indices, batch_size  )

        for i in range(len(minibatches)):
            #Only train generator if n_critic divides i
            train_gen = ( i%n_critic == 0 )
            
            real_images = crop_images( minibatches[i])
            sample_size = real_images.shape[0]  #last mini batch may have different size
            noise = tf.random.normal([sample_size, noise_dim])
               
            update_models( gen_model, disc_model, gen_optimizer, disc_optimizer,
                          real_images, noise, sample_size, lambd, train_gen )
                
        # Save the model every epoch
        gen_checkpoint.save(file_prefix = gen_checkpoint_prefix)
        disc_checkpoint.save(file_prefix = disc_checkpoint_prefix)
        generate_and_save_images(gen_model, epoch + prev_epochs, noise_dim, tag)
        gen_loss, wass_loss = sample_gen_and_wass_losses(gen_model, disc_model, num_images, noise_dim)
        print('Generator loss: ' + str(gen_loss))
        print('Wasserstein loss: ' + str(wass_loss)) 
        print('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
        current_time = datetime.datetime.now()
        print('Time: '+current_time.strftime("%I:%M:%S %p"))
   
    return gen_model, disc_model