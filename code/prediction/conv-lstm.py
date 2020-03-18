# Author: Meera Desai
# 3-18-20

from tensorflow.keras.layers import Lambda, Reshape, Dropout, Permute, Input, add, Conv2D, GaussianNoise, ConvLSTM2D
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tensorboard.plugins.beholder import Beholder 
import tensorflow as tf
import io

dataset = Dataset(r"../../data/omega_redo.nc")

width = len(dataset['/Omega/latitude'][:])
height = len(dataset['/Omega/longitude'][:])

n_input_frames = 5
n_output_frames = 1
input_height = height
input_width = width 
input_depth = 1 # ocean depth /number of depth levels
batch_size = 32
num_epochs = 25
steps_per_epoch = 33
validation_steps = 7
log_dir = "../../figures/conv-lstm/example"

def organize_sequences(img_dir, n_in, n_out, width, height, depth):
# creates output tensor/(numpy array) of shape
# n_samples x n_in+n_out x width x height x depth where output are last in stack
	if img_dir == 'train':
		start_index = 0 
		end_index = 1094 # last index of 2013
	elif img_dir == 'test':
		start_index = 1095 #2014
		end_index = 1355#-6 # to 2018 #save last for pred on batch
	elif img_dir == 'pred':
		start_index = 1355-batch_size*4 # takes some samples from validation data
		end_index = 1355

	all_data = dataset['/Omega/w'][start_index:end_index,:,:,0:depth]
	n_samples = all_data.shape[0]-(n_in+n_out)

	print("found {} {} samples".format(n_samples, img_dir))
	out_data = np.zeros((n_samples,n_in+n_out,width,height,depth))
	for i in range (0,n_samples):
		start = i
		end = i+n_in+n_out
		out_data[i,:,:,:,:] = all_data[start:end,:,:,:]
	return out_data

def organize_sequences_no_overlap(img_dir, n_in, n_out, width, height, depth):
# creates output tensor/(numpy array) of shape
# n_samples x n_in+n_out x width x height x depth where output are last in stack
	if img_dir == 'train':
		start_index = 0 
		end_index = 1094 # last index of 2013
	elif img_dir == 'test':
		start_index = 1095 #2014
		end_index = 1355#-6 # to 2018 #save last for pred on batch
	elif img_dir == 'pred':
		start_index = 1355-batch_size*6 # 
		end_index = 1355

	all_data = dataset['/Omega/w'][start_index:end_index,:,:,0:depth]
	n_samples = all_data.shape[0]//(n_in+n_out) 

	print("found {} {} samples".format(n_samples, img_dir))

	out_data = np.zeros((n_samples,n_in+n_out,width,height,depth))

	j = 0
	for i in np.arange (0,n_samples*(n_in+n_out),(n_in+n_out)):
		start = i
		end = i+n_in+n_out

		out_data[j,:,:,:,:] = all_data[start:end,:,:,:]
		j+=1
	return out_data


def return_batch_for_pred(out_data, batch_number, n_in, flag = 0):

	n_samples = out_data.shape[0]
	
	n_out = out_data.shape[1]-n_in
	depth = out_data.shape[4]
	counter = 0 # counts the number of batches
	np.random.shuffle(out_data)
	n_batches = n_samples // batch_number

	
	if counter >= n_batches:
		counter = 0 
		np.random.shuffle(out_data)
	batch_in = np.zeros((batch_size,n_in,width,height,depth))
	batch_out = np.zeros((batch_size,width,height,depth))

	for j in np.arange(0,(n_samples//batch_size)*batch_size,batch_size):
		
		batch_of_all_data = out_data[j:j+batch_size,:,:,:,:]
		batch_in[:,:,:,:,:] = batch_of_all_data[:,0:n_in,:,:,:]

		batch_out[:,:,:,:] = batch_of_all_data[:,n_out:n_out+1,:,:,:].reshape(batch_size,width,height,depth)
		counter +=1
	return (batch_in,batch_out)

def data_generator(out_data, batch_number, n_in, flag = 0):

	n_samples = out_data.shape[0]
	
	n_out = out_data.shape[1]-n_in
	depth = out_data.shape[4]
	counter = 0 # counts the number of batches
	np.random.shuffle(out_data)
	n_batches = n_samples // batch_number

	while True:
		if counter >= n_batches:
			counter = 0 
			np.random.shuffle(out_data)
		batch_in = np.zeros((batch_size,n_in,width,height,depth))
		batch_out = np.zeros((batch_size,width,height,depth))

		for j in np.arange(0,(n_samples//batch_size)*batch_size,batch_size):
			
			batch_of_all_data = out_data[j:j+batch_size,:,:,:,:]
			batch_in[:,:,:,:,:] = batch_of_all_data[:,0:n_in,:,:,:]

			batch_out[:,:,:,:] = batch_of_all_data[:,n_out,:,:,:].reshape(batch_size,width,height,depth)
			counter +=1

		
		yield (batch_in,batch_out)

def make_plot(prediction, true_label):
	fig, axes = plt.plot(1,2)
	axes[0,0].imshow(prediction.reshape(76,23,3))
	axes[0,0].title("prediction")
	axes[0,1] = plt.inshow(true_label.reshape(76,23,3))
	axes[0,1].title("ground truth")
	return fig

def log_predictions(epoch, logs):
	file_writer = tf.summary.create_file_writer(log_dir)

	pred_data_imp = organize_sequences('pred',n_input_frames,n_output_frames,width,height,input_depth)
	pred_data = return_batch_for_pred(pred_data_imp,32,n_input_frames,flag=1)
	in_ = pred_data[0] #number of samples (1) x n_in x width x height x depth
	out_ = pred_data[1]

	test_pred_raw = (model.predict_on_batch(in_,))

	pred_to_keep = tf.reshape(test_pred_raw[0,:,:,:],(1,76,23,input_depth))
	pred_to_keep = pred_to_keep.numpy()
	tensor_out = tf.concat([out_,pred_to_keep],0)

	#plotting stuff
	top = cm.get_cmap('Blues_r', 128)
	bottom = cm.get_cmap('Reds', 128)
	newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
	newcmp = ListedColormap(newcolors, name='RedBlue')
	_min = -10
	_max = 10
	fig, axes = plt.subplots(2,6)
	for i in range(0,n_input_frames):
		
		p = axes[0,i].pcolormesh(in_[0,i,:,:,:].reshape(76,23),cmap = newcmp,vmin = _min, vmax = _max)
		axes[0,i].set_title("week " +str(i+1),fontsize=7)
		axes[1,i].axis('off')
	axes[0,n_input_frames].pcolormesh(out_[0,:,:,:].reshape(76,23),cmap = newcmp,vmin = _min, vmax = _max)
	axes[0,n_input_frames].set_title("week 6\n(ground truth)",fontsize=7)
	axes[1,n_input_frames].pcolormesh(pred_to_keep.reshape(76,23),cmap = newcmp,vmin = _min, vmax = _max)
	axes[1,n_input_frames].set_title("week 6\n(prediction)",fontsize=7)
	cbaxes = fig.add_axes([.92, 0.1, 0.02, 0.4]) 
	cbar = fig.colorbar(p, label='m/day',cax = cbaxes)
	in_out = plot_to_image(fig)
	
	with file_writer.as_default():
		tf.summary.image("in/out", in_out, step = epoch)



inp = Input((n_input_frames,input_width,input_height,input_depth))

conv_lstm_output_1 = ConvLSTM2D(6, (6,6), padding='same',dropout=0.2)(inp)
conv_output = Conv2D(input_depth, (3,3), padding="same")(conv_lstm_output_1)

model = Model(inputs = [inp],outputs = [conv_output])
model.compile(optimizer='adam', loss='mse')
model.summary()

val_data = organize_sequences("test",n_input_frames,n_output_frames,width,height,input_depth)
train_data = organize_sequences("train",n_input_frames,n_output_frames,width,height,input_depth)



def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


cm_callback = callbacks.LambdaCallback(on_epoch_end = log_predictions)

tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_layer_names=conv_output, write_images= True)
a = model.fit_generator(data_generator(train_data,batch_size,n_input_frames),
                    steps_per_epoch=steps_per_epoch,
                    epochs=num_epochs, callbacks=[tensorboard_callback,cm_callback],
    validation_steps=validation_steps,
    validation_data=data_generator(val_data,batch_size,n_input_frames))





