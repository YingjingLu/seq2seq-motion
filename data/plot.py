import matplotlib.pyplot as plt 
import numpy as np 

# a = np.load( "t.npy" )
# c = a[ 0,0,: ]
# c = c*255.
# print(c[:,:30])
# plt.imshow( c )
# plt.show()

from img_util import *
a = np.expand_dims( np.load( "t_5.npy" ), axis = 5)
print( a.shape )
c = np.expand_dims( a[ 0, :25,:,: ,:], axis = 0)
print( c.shape )
save_image( "t", c )