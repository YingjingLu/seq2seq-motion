import numpy as np 
import gc
import skvideo.io as svio 
import os
import cv2 as cv
gc.enable()

def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

class Data_Source( object ):

	def __init__( self, config ):
		
		self.config = config
		self.init()


	def init( self ):
		os.mkdir( "processed" ) if not os.path.exists( "processed" ) else print( "Processed folder exists" )
		self.cur_file = 0
		# self.load_file()
		self.load_existing()
		# self.data = np.expand_dims( np.load( self.config.data_path ), axis = 5 )
		# np.random.shuffle( self.data )
		self.num_data = 9360
		self.cur_data_index = 0
		
		
		gc.collect()

	def load_existing( self ):
		self.data = np.load( "processed/" + str( self.cur_file ) + ".npy" )
		print("loaded data")
		self.total_len_dict = np.load( "processed/len.npy" )
		self.cur_len_dict = np.zeros_like( self.total_len_dict )
		print("loaded ken")
		self.label = np.load( "processed/class.npy" )
		print( "loaded class" )


	def load_file( self ):
		self.cur_len_dict = list()
		self.total_len_dict = list()
		self.data = []
		self.label = []
		base_path = self.config.data_path 
		if self.config.train:
			file_path = self.config.train_file_list
		else:
			faile_path = self.config.test_file_list
		train_f = open( file_path, "r" )
		line = train_f.readline()
		count = 0
		prev = 0
		self.num_data = 0
		inner_count = 0
		cache = np.zeros( ( self.config.batch_size * self.config.save_batch, self.config.max_frame, self.config.ae_in_h, self.config.ae_in_w, self.config.ae_in_c ), dtype = np.float32 )
		while line != "":
			self.num_data += 1
			[ file_name, class_ ] = line.split( " " )
			total_path = base_path + "/UCF101" + "/" + file_name
			video = self.read_video( total_path )
			print(video.shape)
			self.cur_len_dict.append( 0 )
			self.total_len_dict.append( video.shape[ 0 ] )
			cur_data = np.expand_dims( video, axis = 0 )
			self.label.append( class_ )
			print( "Reading {} succeed".format( file_name ) )
			cache[ inner_count, :cur_data.shape[ 1 ], :, :, : ] = cur_data
			line = train_f.readline()
			count += 1
			inner_count += 1
			if count != 0 and count % ( self.config.batch_size * self.config.save_batch ) == 0:

				np.save( "processed/" + str( prev ), cache )
				gc.collect()
				inner_count = 0
				prev += 1
				cache = None
				gc.collect()
				cache = np.zeros( ( self.config.batch_size * self.config.save_batch, self.config.max_frame, self.config.ae_in_h, self.config.ae_in_w, self.config.ae_in_c ), dtype = np.float32 )
				gc.collect()
			gc.collect()
		np.save( "processed/class.npy", np.array( self.label ) )
		np.save( "processed/len.npy", np.array( self.total_len_dict ) )
		self.data = np.load( "processed/" + str( self.cur_file ) + ".npy" )

	def read_video( self, path, max_frames = 0 ):
		cap = cv.VideoCapture(path)
		frames = []
		try:
			while True:
				ret, frame = cap.read()
				if not ret:
					break
				frame = crop_center_square(frame)
				frame = cv.resize(frame, (self.config.ae_in_w, self.config.ae_in_h) )
				frame = frame[:, :, [2, 1, 0]]
				frames.append( frame )
				
				if len(frames) == max_frames:
					break
		finally:
			cap.release()
		return (np.array(frames) / 255.0 ) [:self.config.max_frame]

	def get_train_batch_ae( self, batch_size = -1, frame_len = -1 ):

		if batch_size == -1:
			batch_size = self.config.batch_size 
		if frame_len == -1:
			frame_len = self.config.frame_len
		
		if self.cur_data_index + batch_size >= self.num_data:
			self.cur_data_index = 0
			print( "----- pass one iteration ----" )
		batch = np.zeros( ( self.config.batch_size, self.config.trans_seq_l, self.config.ae_in_h, self.config.ae_in_w, self.config.ae_in_c ), dtype = np.float32 )
		# print( "lol", batch.shape )
		for i in range( self.config.batch_size ):
			cur_index = self.cur_data_index + i
			cur_frame = self.cur_len_dict[ cur_index ]
			cur_total_frame = self.total_len_dict[ cur_index ]

			converted_index = cur_index - self.cur_file * ( self.config.batch_size * self.config.save_batch )
			data = self.data[ converted_index, cur_frame: cur_frame + frame_len, :, :, : ]
			data = np.expand_dims( data, axis = 0 )
			batch[ i ] = data
			if cur_frame + 2 + frame_len > cur_total_frame:
				self.cur_len_dict[ cur_index ] = 0
			else:
				self.cur_len_dict[ cur_index ] = cur_frame + 2
			# print( "hahahah" )

		self.cur_data_index += batch_size
		if self.cur_data_index > ( ( self.cur_file + 1 ) * ( self.config.batch_size * self.config.save_batch ) - 1 ):
			self.cur_file += 1
			if self.cur_file == 8:
				self.cur_file = 0
			print( "reload data" )
			self.data = None
			gc.collect()
			self.data = np.load( "processed/" + str( self.cur_file ) + ".npy" )
			print( "success reload" )
		# print( batch.shape )
		comp = np.zeros_like( batch[:, self.config.ae_seq_l:, :, :, :]  )
		# print("comp", comp.shape)
		gc.collect()
		return np.concatenate( ( batch[:, :self.config.ae_seq_l, :, :, :], comp ), axis = 1 ), batch
		
		

		
		


		

		

		


