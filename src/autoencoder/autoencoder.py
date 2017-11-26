#use "python3 autoencoder.py -mn basic_ae -s locomotion_small" to test

import sys
sys.path.append('../Lib')
sys.path.append('../Config')
sys.path.append('../utils')
from logger import Logger

from bvhHandler import readBVH,getVelocities
from readBVH import switch_rootrotn2_YZX

from os.path import join,isdir,isfile
from os import listdir

import argparse
import numpy as np
##################################################################
parser = argparse.ArgumentParser(description='inputs to the file')
parser.add_argument("--input_dir","-id",type=str,default="../../dataset/")
parser.add_argument("--subjects","-s",type=str,default="locomotion")
parser.add_argument("--epochs","-e",type=int,default=200)
parser.add_argument("--learning-rate","-lr",type=int,default=0.01)
parser.add_argument("--model-name","-mn",type=str,required = True)
parser.add_argument("--look-around","-la",type=int,default = 5)
parser.add_argument("--batch-size","-bs",type=int,default = 128)

args = parser.parse_args()

subjects = args.subjects
input_dir = args.input_dir
epochs = args.epochs
learning_rate = args.learning_rate
model_name = args.model_name
look_around = args.look_around
batch_size = args.batch_size

motion_classes = getattr(__import__(subjects,fromlist=["motion_classes"]),"motion_classes")
##################################################################
logger = Logger("autoencoder")
##################################################################
#Data generation

def read_files(dir,subjects):
	files = []
	for subject in subjects:
		files+=[join(join(dir,subject),tmp) for tmp in listdir(join(dir,subject))]
	logger.logger.info("Files to be read %d", len(files))
	data = []
	for file in files:
		data.append(readBVH(file))
	logger.logger.info("Read %d files", len(files))
	return data

data = read_files(input_dir,motion_classes)
data = switch_rootrotn2_YZX(data)
logger.logger.info("Rotation swapped")
for datum in data:
	datum = getVelocities(datum,1)

channels = len(data[0][0])

data_in_format = []
for datum in data:	
	for i in range(len(datum)-look_around+1):
		data_in_format.append(np.array(datum[i:i+look_around]))
npdata = np.array(data_in_format)
print(npdata.shape)

##################################################################
#Training
from network import AutoEncoder

autoencoder = AutoEncoder(join("model",model_name),1,learning_rate,(look_around,channels))

for i in range(epochs):
	for j in range(0,npdata.shape[0],batch_size):
		autoencoder.run_training(npdata[j:min(j+batch_size,npdata.shape[0])])
		loss = autoencoder.get_loss(npdata[j:min(j+batch_size,npdata.shape[0])])
		logger.logger.info("Epoch: %d/%d | subiteration %d/%d | minibatch loss: %f",i+1,epochs,j+1,npdata.shape[0],loss)
	autoencoder.save_model()


