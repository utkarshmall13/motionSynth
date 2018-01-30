#use "python3 autoencoder.py -mn basic_ae -s locomotion_small" to test

import sys
sys.path.append('../Lib')
sys.path.append('../Config')
sys.path.append('../utils')
from logger import Logger

import spatialbvhHandler as sbh
from readBVH import switch_rootrotn2_YZX

from os.path import join,isdir,isfile
from os import listdir

import argparse
import numpy as np
##################################################################
parser = argparse.ArgumentParser(description='inputs to the file')
parser.add_argument("--input_dir","-id",type=str,default="../../dataset/spatial/")
parser.add_argument("--subjects","-s",type=str,default="locomotion")
parser.add_argument("--epochs","-e",type=int,default=200)
parser.add_argument("--learning-rate","-lr",type=float,default=0.01)
parser.add_argument("--model-name","-mn",type=str,required = True)
parser.add_argument("--look-around","-la",type=int,default = 24)
parser.add_argument("--window-size","-ws",type=int,default = 120)
parser.add_argument("--batch-size","-bs",type=int,default = 128)
parser.add_argument("--keep-prob","-kp",type=float,default = 0.5)

args = parser.parse_args()

subjects = args.subjects
input_dir = args.input_dir
epochs = args.epochs
learning_rate = args.learning_rate
model_name = args.model_name
look_around = args.look_around
window_size = args.window_size
batch_size = args.batch_size
keep_prob = args.keep_prob

motion_classes = getattr(__import__(subjects,fromlist=["motion_classes"]),"motion_classes")
##################################################################
logger = Logger("conv_autoencoder")
##################################################################
#Data generation

def read_files(dir,subjects):
	files = []
	for subject in subjects:
		files+=[join(join(dir,subject),tmp) for tmp in listdir(join(dir,subject))]
	logger.logger.info("Files to be read %d", len(files))
	data = []
	for file in files:
		datum = sbh.readBVH(file)
		datum = sbh.represent20(datum)
		datum = sbh.getOrientation(datum)
		datum = sbh.subsample(datum)
		for i in range(4):
			datum[i] = sbh.convert_to_relative(datum[i])
		data+=datum
	mean,std = sbh.find_mean_std(data)
	data = sbh.normalize(data,mean,std)

	logger.logger.info("Read %d files", len(files))
	return data

data = read_files(input_dir,motion_classes)

channels = len(data[0][0])

data_in_format = []
for datum in data:
	for i in range(len(datum)-window_size+1):
		data_in_format.append(np.array(datum[i:i+window_size]))
npdata = np.array(data_in_format)
logger.logger.info("Formatted data with size %d",npdata.shape[0])

##################################################################
#Training
from network import ConvAutoEncoder

autoencoder = ConvAutoEncoder(join("model",model_name),keep_prob,learning_rate,(window_size,channels),look_around)

for i in range(epochs):
	for j in range(0,npdata.shape[0],batch_size):
		autoencoder.run_training(npdata[j:min(j+batch_size,npdata.shape[0])])
		# logger.logger.info("Epoch: %d/%d | subiteration %d/%d | minibatch loss: %f",i+1,epochs,j+1,npdata.shape[0],loss)
	Loss = []
	for j in range(0,npdata.shape[0],batch_size):
		loss = autoencoder.get_loss(npdata[j:min(j+batch_size,npdata.shape[0])])
		Loss.append(loss)
	logger.logger.info("Epoch: %d/%d | total loss: %f",i+1,epochs,sum(Loss)/len(Loss))
	autoencoder.save_model()


