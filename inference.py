import math
import os
import sys
import h5py
import math
import numpy as np

from keras.datasets import mnist


# The following two functions from
# https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python

def print_hdf5_file_structure(file_name) :
    """Prints the HDF5 file structure"""
    file = h5py.File(file_name, 'r') # open read-only
    item = file #["/Configure:0000/Run:0000"]
    print_hdf5_item_structure(item)
    file.close()
 
def print_hdf5_item_structure(g, offset='    ') :
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    if   isinstance(g,h5py.File) :
        print g.file, '(File)', g.name
 
    elif isinstance(g,h5py.Dataset) :
        print '(Dataset)', g.name, '    len =', g.shape #, g.dtype
 
    elif isinstance(g,h5py.Group) :
        print '(Group)', g.name
 
    else :
        print 'WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name
        sys.exit ( "EXECUTION IS TERMINATED" )
 
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        for key,val in dict(g).iteritems() :
            subg = val
            print offset, key, #,"   ", subg.name #, val, subg.len(), type(subg),
            print_hdf5_item_structure(subg, offset + '    ')
            

################
## Look at h5
################

h5File = h5py.File('my_model_weights.h5')

# Print h5 contents
#for item in h5File.attrs.keys():
#    print(item + ":", h5File.attrs[item])

print_hdf5_file_structure('my_model_weights.h5')
    

##################################
## Get weights, biases, and data
##################################

#Get kernels/filters/weights and biases
conv_k = h5File['/conv2d_1/conv2d_1/kernel:0'][()]
conv_b = h5File['/conv2d_1/conv2d_1/bias:0'][()]
dense_k = h5File['/dense_1/dense_1/kernel:0'][()]
dense_b = h5File['/dense_1/dense_1/bias:0'][()]


(x_train, y_train), (x_test, y_test) = mnist.load_data()

### Get a sample
#print x_train.shape
#print y_train.shape
x_sample = x_train[0,:,:]
y_sample = y_train[0]

in_height = 28;
in_width  = 28;
in_chann  = 1; 

f_height   = 3;
f_width    = 3;
f_outchann = 4;         #number of filters  

stride_width = 1;
stride_height = 1;
padding = "same";

# Derived

f_inchann  = in_chann;  #number of input channels


# Padding

out_width  = int(math.ceil(float(in_width) / float(stride_width)))
if (in_width % stride_width == 0):
    pad_along_width = max(f_width - stride_width, 0)
else:
    pad_along_width = max(f_width - (in_width % stride_width), 0)
pad_left = pad_along_width // 2
pad_right = pad_along_width - pad_left

out_height  = int(math.ceil(float(in_height) / float(stride_height)))
if (in_height % stride_height == 0):
    pad_along_height = max(f_height - stride_height, 0)
else:
    pad_along_height = max(f_height - (in_height % stride_height), 0)
pad_top = pad_along_height // 2
pad_bottom = pad_along_height - pad_top

print("pad_left: {}".format(pad_left)) 
print("pad_right: {}".format(pad_right)) 
print("pad_top: {}".format(pad_top)) 
print("pad_bottom: {}".format(pad_bottom)) 

if padding == "same":
    in_width = in_width + pad_left + pad_right
    in_height = in_height + pad_top + pad_bottom

print("in_width, post padding, should be: {}".format(in_width))
print("in_height, post padding, should be: {}".format(in_height))


#f1 = open('pre.txt', 'w')
#np.savetxt(f1,x_sample)
x_sample = np.pad(x_sample, [(pad_top,pad_bottom),(pad_left,pad_right)], 'constant')
#f2 = open('post.txt', 'w')
#np.savetxt(f2,x_sample)

n_mult = 0
n_add = 0
for ow in range(0, out_width):
    for oh in range(0, out_height):
        for f in range(0, f_outchann):
            channel_sum = 0;
            for c in range(0, in_chann):
                
                #data is f_width*f_height
                #filt is f_width*f_height
                n_mult = n_mult + f_width*f_height
                
                
