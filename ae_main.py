from autoencoder import Autoencoder
from config import Config 
from data_source import Data_Source

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



