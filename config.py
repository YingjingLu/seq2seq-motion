import os
class Config( object ):

	def __init__( self ):

		## Autoencoder configuration ##
		self.ae_in_w = 64
		self.ae_in_h = 64
		self.ae_in_c = 1
		# self.ae_in_l = 16
		self.ae_seq_l = 4
		self.ae_out_seq_l = 5
		self.max_frame = 80
		self.save_batch = 30 

		self.ae_loss = ""
		self.bn = True
		self.encoder_start_dim = 16
		self.decoder_start_dim = 1

		## Transformer Configuration ##
		self.trans_in_w = 4
		self.trans_in_h = 4
		self.trans_in_c = 128
		self.trans_in_l = 16 # max length of a given sequence
		self.trans_seq_l = self.ae_out_seq_l # length of each unit being input at each state

		self.trans_conv_ndim = 3
		self.trans_output_channel = 3 # 3 pictures as a group to be input to advance a state
		self.trans_kernel_shape = [ 3,3,3 ]


		### Training ###
		self.batch_size = 16 # batch size to be used to sample a batch of video frame for training
		self.frame_len = self.ae_out_seq_l # number of frames for each video clip to be passed to training
		self.train_iter = 600000
		self.start_lr = 1e-4
		self.lr_decay = 0.9
		self.lr_decay_every = 30000
		self.ckpt_dir = "mmd_mmd_ae"
		self.res_dir = "ae_seq_4_sp2_res"
		self.save_every = 1000

		self.train = 0
		self.load_model = "mmd_mmd_ae/200000/model.ckpt" #"mmd_mmd_ae/212000/model.ckpt"

		### Data Preprocessing ###
		self.data_source = "mnist"

		if self.data_source == "mnist":
			self.data_path = "../../data/t_2.npy"
		elif self.data_source == "ucf":
			self.data_path = "../../data/UCF101"
			self.train_file_list = "../../data/UCF101/ucfTrainTestlist/trainlist01.txt"
			self.test_file_list = "../../data/UCF101/ucfTrainTestlist/testlist01.txt"
		self.train_test_split = 0.75 # percentage of training samples within the entire dataset
		

	def init( self ):
		os.mkdir( self.ckpt_dir ) if not os.path.exists( self.ckpt_dir ) else print()
		os.mkdir( self.res_dir ) if not os.path.exists( self.res_dir ) else print()



