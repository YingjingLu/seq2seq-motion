import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import math

def unnormalize(img, cdim):
    img_out = np.zeros_like(img)
    for i in range(cdim):
        # img_out[:, :, i] = 255.* ((img[:, :, i] + 1.) / 2.0)
        img_out[:, :, i] = 255.* (np.clip(img[:, :, i], 0, 1))
    img_out = img_out.astype(np.uint8)
    return img_out

def display_color(img, file_start):
    fig = plt.figure(figsize=(1, img.shape[0]))
    # fig = plt.figure()
    gs = gridspec.GridSpec(1,img.shape[0])
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(img):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        k = unnormalize(sample, 3)
        plt.imshow(k)
    plt.savefig(file_start +'.png', bbox_inches='tight')
    plt.close(fig)

def display_grey(img, file_start):
    """
        Args:
            img ( depth, h, w, c )
    """
    img = (img*255.)
    shape = img.shape
    fig = plt.figure( figsize=( 10, shape[ 0 ] * 10 ) )
    gs = gridspec.GridSpec( 1, shape[ 0 ] )
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(img):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(sample.shape[0], sample.shape[1] ), cmap='Greys_r')
    plt.savefig(file_start +'.png', bbox_inches='tight')
    plt.close(fig)

def save_image( file_path, image_matrix ):
    # RGB
    if( image_matrix.shape[-1] == 3):
        for i in range( image_matrix.shape[ 0 ] ):
            display_color( image_matrix[ i, :,:, :, : ], file_path +"/"+ str( i ) )
    # grey scale
    elif ( image_matrix.shape[-1] ) == 1:
        for i in range( image_matrix.shape[ 0 ] ):
            display_grey( image_matrix[ i, :, :, :, : ], file_path + "/" + str( i ) )
    else:
        print( "No such channel color scale" )