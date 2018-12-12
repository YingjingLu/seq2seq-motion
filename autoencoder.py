import tensorflow as tf 
from ae_models import *
import os
import numpy as np
from img_util import *

class Autoencoder( object ):
    
    def __init__( self, config ):
        self.sess = tf.Session()
        self.config = config
        self.add_input_placeholder()
        self.construct_ae()
        self.construct_loss()
        self.construct_optimizer()
        self.init()
        self.saver = tf.train.Saver( max_to_keep = 1000 )

    def init( self ):
        self.sess.run( [ tf.global_variables_initializer() ] )

    def add_input_placeholder( self ):
        with tf.variable_scope( "ae" ) as scope:
            self.sample_input = tf.placeholder( tf.float32, 
                                                [ self.config.batch_size, self.config.ae_out_seq_l, self.config.ae_in_h, 
                                                self.config.ae_in_w, self.config.ae_in_c ], 
                                                name = "ae_input_seq" )
            self.target_input = tf.placeholder( tf.float32, 
                                                [ self.config.batch_size, self.config.ae_out_seq_l, self.config.ae_in_h, 
                                                self.config.ae_in_w, self.config.ae_in_c ], 
                                                name = "ae_input_seq" )
            self.encode_input = tf.placeholder( tf.float32, 
                                                [ self.config.batch_size, self.config.trans_seq_l, self.config.trans_in_h, 
                                                self.config.trans_in_w, self.config.trans_in_c ], 
                                                name = "ae_input_encode" )
            self.lr_input = tf.placeholder( tf.float32, [], name = "lr_input" )

    def construct_ae( self ):
        with tf.variable_scope( "ae" ) as scope:
            if self.config.bn:
                    self.encode = img_encoder3d_batch_norm( self.sample_input, self.config.encoder_start_dim )
                    self.decode, self.logits = img_decoder3d_batch_norm( self.encode, 
                                                self.config.ae_out_seq_l, self.config.ae_in_h, 
                                                 self.config.ae_in_w, self.config.ae_in_c, self.config.batch_size, self.config.decoder_start_dim )
                    self.decode_out, self.logits_out = img_decoder3d_batch_norm( self.encode_input, 
                                                    self.config.ae_out_seq_l, self.config.ae_in_h, 
                                                    self.config.ae_in_w, self.config.ae_in_c, self.config.batch_size, self.config.decoder_start_dim )
            else:
                self.encode = img_encoder3d( self.sample_input, self.config.encoder_start_dim )
                self.decode, self.logits = img_decoder3d( self.encode, 
                                             self.config.ae_out_seq_l, self.config.ae_in_h, 
                                             self.config.ae_in_w, self.config.ae_in_c, self.config.batch_size, self.config.decoder_start_dim )
                self.decode_out, self.logits_out = img_decoder3d( self.encode_input, 
                                                 self.config.ae_out_seq_l, self.config.ae_in_h, 
                                                 self.config.ae_in_w, self.config.ae_in_c, self.config.batch_size, self.config.decoder_start_dim )

    
    def construct_loss( self ):
        self.mse = tf.reduce_mean( tf.square( self.target_input - self.decode ) )
        if self.config.ae_loss == "mmd":
            true_label = tf.random_normal( self.encode.get_shape() )
            mmd_loss = self.compute_mmd( self.encode, true_label )
            self.loss = self.mse + mmd_loss
        else:
            self.loss = self.mse


    def construct_optimizer( self ):
        self.optim = tf.train.AdamOptimizer( self.lr_input, beta1=0.9, beta2 = 0.99 ).minimize( self.loss,
            var_list = tf.trainable_variables( scope = "ae" ) )

    def encode_inference( self, seq ):
        res = self.sess.run( self.encode, feed_dict = { self.encode: seq } )
        return res

    def decode_inference( self, encode ):
        res = self.sess.run( self.decode_out,
                             feed_dict = { self.encode_input: encode } )
        return res

    def train( self, data_source ):
        loss_list = []
        start_iter = 0
        lr = self.config.start_lr
        if self.config.load_model != "":
            start_iter = int( self.config.load_model.split( "/" )[ -2 ] )
            self.saver.restore( self.sess, self.config.load_model )
        print( "=== start training ===" )
        for i in range( start_iter + 1, self.config.train_iter ):
            sample, target = data_source.get_train_batch_ae()
            # print( "sample", sample.shape )
            self.sess.run( [ self.optim ], feed_dict = { self.sample_input: sample, self.lr_input: lr, self.target_input: target } )

            if i % self.config.save_every == 0:
                loss, recon = self.sess.run( [ self.mse, self.decode ], feed_dict = { self.sample_input: sample, self.target_input: target } )
                loss_list.append( loss )
                print( "Iteration {} mse {}".format( i, loss ) )

                img_path = self.config.res_dir + "/" + str( i )
                os.mkdir( img_path ) if not os.path.exists( img_path ) else print()
                os.mkdir( img_path+"/true" ) if not os.path.exists( img_path+"/true" ) else print()
                os.mkdir( img_path+"/res" ) if not os.path.exists( img_path+"/res" ) else print()
                save_image( img_path+"/true", recon )
                save_image( img_path+"/res", target )

                save_path = self.config.ckpt_dir + "/" + str( i )
                os.mkdir( save_path ) if not os.path.exists( save_path ) else print()
                self.saver.save( self.sess, save_path + "/" + "model.ckpt" )
            if i > 0 and i % self.config.lr_decay_every == 0:
                lr = lr * self.config.lr_decay

        np.save( "loss.npy", np.array( loss_list ) )

    def test( self, inputs, length, stride = 1 ):
        # recursively ggenerate the next frame for length long
        ret = []
        pred = self.sess.run( self.decode, feed_dict = { self.sample_input: inputs } ) # ( batch, length, width, height, channel )
        ret.append( pred )
        length -= 1
        while length > 0:
            pred = self.sess.run( self.decode, feed_dict = { self.sample_input: pred[ :, stride:, :, :, : ] } )
            ret.append( np.expand_dims( pred[ :, -1, :, :, : ], axis = 1 ) )
        
        return np.concatenate( ret, axis = 1 )

    def calc_pred_mse( self, seq_m ):
        length = seq_m.shape[ 1 ] - self.config.ae_seq_l
        inputs = seq_m[ :, :self.config.ae_seq_l, :, :, : ]
        ret = self.test( inputs, length )
        truth = seq_m[ :, self.config.ae_seq_l:, :, :, : ]
        return np.mean( np.square( truth - ret ) )

    def compute_mmd( self, x, y, sigma_sqr=1.0):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

    def compute_kernel( self, x, y ):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, x.shape[ 1 ], x.shape[2], x.shape[3], x.shape[4]])), tf.stack([1, y_size, 1, 1, 1, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, x.shape[ 1 ], x.shape[2], x.shape[3], x.shape[4]])), tf.stack([x_size, 1, 1, 1, 1, 1]))
        return tf.exp( -tf.reduce_mean(tf.square( tiled_x - tiled_y ), axis=[2] ) )
    
    

    
    