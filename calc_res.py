from autoencoder import Autoencoder
from config import Config 
from data_source import Data_Source
import numpy as np 
import tensorflow as tf 


config = Config()
config.init()
model = Autoencoder( config )
if config.data_source == "mnist":
	from data_source import Data_Source
elif config.data_source == "ucf":
	from data_source_ucf import Data_Source
elif config.data_source == "youtube":
	from data_source_youtube import Data_Source

data_source = Data_Source( config )
if config.train:
    model.train( data_source )
else:
    assert config.load_model != ""
    model.saver.restore( model.sess, config.load_model )

    sample, target = data_source.get_train_batch_ae()
    logit, decode = model.sess.run( [ model.logits, model.decode ], feed_dict = { model.sample_input: sample } )

    print( "=== cross_entropy ===" )

    ce = model.sess.run( tf.reduce_mean( tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits( logits = logit, labels = target ), axis = [2,3,4] ) ))
    print( ce )
    print()

    print( "=== MSE ===" )

    overall = np.mean( np.square( decode - target ) )
    print( "Overall MSE: ", overall )
    print("")

    recon = decode[ :, :config.ae_seq_l, :, :, : ]
    truth = sample[ :, :config.ae_seq_l, :, :, : ]
    recon_mse = np.mean( np.square( recon - truth ) )
    print( "recon mse: ", recon_mse )
    print( "" )

    mse_list = []
    for i in range( config.ae_out_seq_l - config.ae_seq_l ):
    	pred = decode[ :, i + config.ae_seq_l, :, :, : ]
    	truth = target[ :, i + config.ae_seq_l, :, :, : ]

    	mse = np.mean( np.square( pred - truth ) )
    	mse_list.append( mse )

    print( "mse decay", mse_list )
    print()

    print( "=== SSIM ===" )






