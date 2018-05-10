import math
import os
import sys
import h5py
import math
import numpy as np

import keras
from keras import backend as K
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

# Print h5 contents
#for item in h5File.attrs.keys():
#    print(item + ":", h5File.attrs[item])

h5File = h5py.File('my_model_weights.h5')
print_hdf5_file_structure('my_model_weights.h5')
    

##################################
## Get weights, biases, and data
##################################

#Get kernels/filters/weights and biases
conv_k = h5File['/conv2d_1/conv2d_1/kernel:0'][()]
conv_b = h5File['/conv2d_1/conv2d_1/bias:0'][()]
dense_k = h5File['/dense_1/dense_1/kernel:0'][()]
dense_b = h5File['/dense_1/dense_1/bias:0'][()]

print "conv_k shape:",conv_k.shape

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print 'x_train shape: ',x_train.shape
print 'y_train shape: ',y_train.shape
print 'train samples: ',x_train.shape[0]
print 'test samples: ',x_test.shape[0]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


### Get a sample
x_sample = x_train[0,:,:,:]
y_sample = y_train[0]
print 'x_sample shape: ',x_sample.shape
print 'y_sample shape: ',y_sample.shape

f0 = open('inference_input.txt', 'w')
np.savetxt(f0,x_train[0:1,:,:,:].flatten())

in_height = 28;
in_width  = 28;
in_chann  = 1; 

f_height   = 3;
f_width    = 3;
f_outchann = 1; #number of filters  

stride_width = 1;
stride_height = 1;
padding = "same";

# Derived

f_inchann  = in_chann;  #number of input channels
n_filters  = f_outchann;

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
    x_sample = np.pad(x_sample, [(pad_top,pad_bottom),(pad_left,pad_right),(0,0)], 'constant')
    print "x_sample shape: ",x_sample.shape
    #f2 = open('post.txt', 'w')
    #np.savetxt(f2,x_sample)


conv_out = np.zeros((out_height,out_width,n_filters))


n_mult = 0
n_add = 0
for oh in range(0, out_height):
    for ow in range(0, out_width):
        for f in range(0, f_outchann): #n_filters
            channel_sum = 0;
            for c in range(0, in_chann):

                #count multiplications
                n_mult = n_mult + f_width*f_height

                #get filter
                my_filter = conv_k[:,:,c,f]
                
                #select data
                x_buffer = x_sample[:,:,c]
                x_buffer = x_buffer[oh*stride_height:oh*stride_height+f_height,ow*stride_width:ow*stride_width+f_width]

                #do multiplication
                my_mult = np.multiply(x_buffer, my_filter);
                
                #sum
                my_dot = np.sum(my_mult)
                channel_sum += my_dot

                if ow==0 and oh==0 and f==0 and c==0:
                    #if np.sum(x_buffer)>0 :
                    print "buffer shape: ",x_buffer.shape
                    print "filter shape: ",my_filter.shape
                    print "mult shape: ",my_mult.shape
                    print "dot shape: ",my_dot.shape
                    print "channel sum shape: ",channel_sum.shape
                    print "buffer : ",x_buffer
                    print "filter : ",my_filter
                    print "mult : ",my_mult
                    print "dot : ",my_dot
                    print "channel sum : ",channel_sum

            print "conv_b[f] ",conv_b[f]
            conv_out[oh,ow,f] = channel_sum + conv_b[f]
            #print "conv_out[oh,ow,f] ",conv_out[oh,ow,f]


print "conv_out shape: ",conv_out.shape

#Rest of network
conv_out = conv_out * (conv_out > 0) #relu

f3 = open('inference.txt', 'w')
np.savetxt(f3, conv_out[:,:,0].flatten())

conv_out = conv_out.flatten()
print "flattened shape: ",conv_out.shape
dnn_out = np.dot(conv_out, dense_k)+dense_b
dnn_out = np.exp(dnn_out) / sum(np.exp(dnn_out)) #softmax
print "Network output: ",dnn_out


#Only the Conv2d part of Keras

n_kernels = 1

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

modelc = Sequential()
modelc.add(Conv2D(n_kernels, kernel_size=(3, 3),
                 activation='relu',padding='same',
                 input_shape=input_shape))
modelc.load_weights('my_model_weights.h5',by_name=True)
print("conv_out shape: ",modelc.predict(x_train[0:1]).shape)

fc = open('model.txt', 'w')
np.savetxt(fc, modelc.predict(x_train[0:1])[0,:,:,0].flatten())

modelc.save_weights('my_modelc_weights.h5')
outfile = open('modelc.json','wb')
jsonString = modelc.to_json()
import json
with outfile:
    obj = json.loads(jsonString)
    json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
    outfile.write('\n')

h5File = h5py.File('my_modelc_weights.h5')
print_hdf5_file_structure('my_modelc_weights.h5')

