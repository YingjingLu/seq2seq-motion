import tensorflow as tf 
from dc_gan_util import conv3d, deconv3d, batch_norm, conv2d, deconv2d, conv_out_size_same, batch_norm

def img_encoder3d( inputs, start_filter_num = 16 ):
    with tf.variable_scope( "img_encoder3d", reuse = tf.AUTO_REUSE ) as scope:

        print( "--- encoder ---" )
        conv = tf.nn.relu( conv3d( inputs, start_filter_num, name = "conv1a" ) )
        # conv = tf.nn.max_pool3d( conv, [ 1, 1, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], "SAME", name = "pool1" )

        print( "l1 shape", conv.get_shape().as_list() )

        conv = tf.nn.relu( conv3d( conv, start_filter_num * 2, name = "conv2a" ) )
        # conv = tf.nn.max_pool3d( conv, [ 1, 2, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], "SAME", name = "pool2" )

        conv = tf.nn.relu( conv3d( conv, start_filter_num * 4, name = "conv3a" ) )
        # conv = tf.nn.relu( conv3d( conv, start_filter_num * 4, name = "conv3b" ) )
        # conv = tf.nn.max_pool3d( conv, [ 1, 2, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], "SAME", name = "pool3" )

        conv = tf.nn.relu( conv3d( conv, start_filter_num * 8, name = "conv4a" ) )
        conv = tf.nn.relu( conv3d( conv, start_filter_num * 8, name = "conv4b" ) )
        # conv = tf.nn.max_pool3d( conv, [ 1, 2, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], "SAME", name = "pool4" )

        print( "output", conv.get_shape().as_list() )
        return conv

def img_encoder3d_batch_norm( inputs, start_filter_num = 16 ):
    with tf.variable_scope( "img_encoder3d", reuse = tf.AUTO_REUSE ) as scope:
        bn_1 = batch_norm(name='img_ebn_1')
        bn_2 = batch_norm(name='img_ebn_2')
        bn_3 = batch_norm(name='img_ebn_3')
        bn_4 = batch_norm(name='img_ebn_4')
        bn_5 = batch_norm(name='img_ebn_5')

        print( "--- encoder ---" )
        conv = tf.nn.relu( bn_1( conv3d( inputs, start_filter_num, name = "conv1a" ) ) )
        # conv = tf.nn.max_pool3d( conv, [ 1, 1, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], "SAME", name = "pool1" )

        print( "l1 shape", conv.get_shape().as_list() )

        conv = tf.nn.relu( bn_2( conv3d( conv, start_filter_num * 2, name = "conv2a" ) ) )
        # conv = tf.nn.max_pool3d( conv, [ 1, 2, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], "SAME", name = "pool2" )

        conv = tf.nn.relu( bn_3( conv3d( conv, start_filter_num * 4, name = "conv3a" ) ) )
        # conv = tf.nn.relu( conv3d( conv, start_filter_num * 4, name = "conv3b" ) )
        # conv = tf.nn.max_pool3d( conv, [ 1, 2, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], "SAME", name = "pool3" )

        conv = tf.nn.relu( bn_4( conv3d( conv, start_filter_num * 8, name = "conv4a" ) ) )
        conv = tf.nn.relu( bn_5( conv3d( conv, start_filter_num * 8, name = "conv4b" ) ) )
        # conv = tf.nn.max_pool3d( conv, [ 1, 2, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], "SAME", name = "pool4" )

        print( "output", conv.get_shape().as_list() )
        return conv


def img_decoder3d( inputs, out_depth, out_h, out_w, out_c, batch_size,
                     start_filter_num = 1 ):
    with tf.variable_scope( 'img_decoder3d', reuse = tf.AUTO_REUSE ) as scope:
        sd, sh, sw = out_depth, out_h, out_w
        sd2, sh2, sw2 = out_depth, conv_out_size_same( sh, 2 ),conv_out_size_same( sw, 2 )
        sd4, sh4, sw4 = out_depth, conv_out_size_same( sh2, 2 ),conv_out_size_same( sw2, 2 )
        sd8, sh8, sw8 = out_depth, conv_out_size_same( sh4, 2 ),conv_out_size_same( sw4, 2 )
        sd16, sh16, sw16 = out_depth, conv_out_size_same( sh8, 2 ),conv_out_size_same( sw8, 2 )
        sd32, sh32, sw32 = out_depth, conv_out_size_same( sh16, 2 ),conv_out_size_same( sw16, 2 )

        print( "--- deconv ---" )
        print( "inputs", inputs.get_shape().as_list() )
        deconv = tf.nn.relu( deconv3d( inputs, [ batch_size, sd16, sh16, sw16, start_filter_num * 64 ], name = "deconv0" ) )
        print( "l 0", deconv.get_shape().as_list() )
        deconv = tf.nn.relu( deconv3d( deconv, [ batch_size, sd8, sh8, sw8, start_filter_num * 32 ], name = "deconv1" ) )
        print( "l 1", deconv.get_shape().as_list() )
        deconv = tf.nn.relu( deconv3d( deconv, [ batch_size, sd4, sh4, sw4, start_filter_num * 16 ], name = "deconv2" ) )
        print( "l 2", deconv.get_shape().as_list() )
        deconv = tf.nn.relu( deconv3d( deconv, [ batch_size, sd2, sh2, sw2, start_filter_num * 8 ], name = "deconv3" ) )
        print( "l 3", deconv.get_shape().as_list() )
        deconv =  deconv3d( deconv, [ batch_size, sd, sh, sw, out_c ], name = "deconv4" )
        print( "l 4", deconv.get_shape().as_list() )
        return tf.nn.tanh( deconv ), deconv

def img_decoder3d_batch_norm( inputs, out_depth, out_h, out_w, out_c, batch_size,
                              start_filter_num = 1 ):
    with tf.variable_scope( 'img_decoder3d', reuse = tf.AUTO_REUSE ) as scope:
        bn_1 = batch_norm(name='img_dbn_1')
        bn_2 = batch_norm(name='img_dbn_2')
        bn_3 = batch_norm(name='img_dbn_3')
        bn_4 = batch_norm(name='img_dbn_4')
        bn_5 = batch_norm(name='img_dbn_5')

        sd, sh, sw = out_depth, out_h, out_w
        sd2, sh2, sw2 = out_depth, conv_out_size_same( sh, 2 ),conv_out_size_same( sw, 2 )
        sd4, sh4, sw4 = out_depth, conv_out_size_same( sh2, 2 ),conv_out_size_same( sw2, 2 )
        sd8, sh8, sw8 = out_depth, conv_out_size_same( sh4, 2 ),conv_out_size_same( sw4, 2 )
        sd16, sh16, sw16 = out_depth, conv_out_size_same( sh8, 2 ),conv_out_size_same( sw8, 2 )
        sd32, sh32, sw32 = out_depth, conv_out_size_same( sh16, 2 ),conv_out_size_same( sw16, 2 )

        print( "--- deconv ---" )
        print( "inputs", inputs.get_shape().as_list() )
        deconv = tf.nn.relu( bn_1( deconv3d( inputs, [ batch_size, sd16, sh16, sw16, start_filter_num * 64 ], name = "deconv0" ) ) )
        print( "l 0", deconv.get_shape().as_list() )
        deconv = tf.nn.relu( bn_2( deconv3d( deconv, [ batch_size, sd8, sh8, sw8, start_filter_num * 32 ], name = "deconv1" ) ) )
        print( "l 1", deconv.get_shape().as_list() )
        deconv = tf.nn.relu( bn_3( deconv3d( deconv, [ batch_size, sd4, sh4, sw4, start_filter_num * 16 ], name = "deconv2" ) ) )
        print( "l 2", deconv.get_shape().as_list() )
        deconv = tf.nn.relu( bn_4( deconv3d( deconv, [ batch_size, sd2, sh2, sw2, start_filter_num * 8 ], name = "deconv3" ) ) )
        print( "l 3", deconv.get_shape().as_list() )
        deconv =  deconv3d( deconv, [ batch_size, sd, sh, sw, out_c ], name = "deconv4" )
        print( "l 4", deconv.get_shape().as_list() )
        return tf.nn.tanh( deconv ), deconv