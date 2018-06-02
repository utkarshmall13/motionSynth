#use "python3 autoencoder.py -mn convae -s locomotion_small" to test

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
parser.add_argument("--learning-rate","-lr",type=float,default=0.1)
parser.add_argument("--model-name","-mn",type=str,required = True)
parser.add_argument("--look-around","-la",type=int,default = 24)
parser.add_argument("--window-size","-ws",type=int,default = 120)
parser.add_argument("--batch-size","-bs",type=int,default = 128)
parser.add_argument("--keep-prob","-kp",type=float,default = 0.5)
parser.add_argument("--mode","-m",type=str,required = True)

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
mode = args.mode

motion_classes = getattr(__import__(subjects,fromlist=["motion_classes"]),"motion_classes")
##################################################################
logger = Logger("conv_autoencoder")
##################################################################
#Data generation
# import csv

def read_files(dir,subjects):
	files = []
	for subject in subjects:
		files+=[join(join(dir,subject),tmp) for tmp in listdir(join(dir,subject))]
	logger.logger.info("Files to be read %d", len(files))
	data = []
	for file in sorted(files):
		# print(file)
		datum = sbh.readBVH(file)
		datum = sbh.represent20(datum)
		datum = sbh.getOrientation(datum)
		datum = sbh.subsample(datum)
		for i in range(4):
			datum[i] = sbh.convert_to_relative(datum[i])
			# with open("out.csv","w") as ofd:
			# 	writer = csv.writer(ofd,delimiter=" ")
			# 	for datumm in datum[i]:
			# 		writer.writerow(datumm)
			# 	exit()
		data+=datum
	mean,std = sbh.find_mean_std(data)
	np.save(model_name+"_mean.npy",mean)
	np.save(model_name+"_std.npy",std)
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


if(mode=="continue"):
	autoencoder.restore_model()
if(mode=="train" or mode=="continue"):
	for i in range(epochs):
		for j in range(0,npdata.shape[0],batch_size):
			autoencoder.run_training(npdata[j:min(j+batch_size,npdata.shape[0])],justreg=i<20)
			if(j%10==0):
				loss = autoencoder.get_loss(npdata[j:min(j+batch_size,npdata.shape[0])])
				logger.logger.info("Epoch: %d/%d | subiteration %d/%d | minibatch loss: %s",i+1,epochs,j+1,npdata.shape[0],loss)
		Loss = []
		for j in range(0,npdata.shape[0],batch_size):
			loss = autoencoder.get_loss(npdata[j:min(j+batch_size,npdata.shape[0])],which="main")
			Loss.append(loss)
		logger.logger.info("Epoch: %d/%d | total loss: %f",i+1,epochs,sum(Loss)/len(Loss))
		autoencoder.save_model()
import csv
if(mode=="test"):
	mean = np.load(model_name+"_mean.npy")
	std = np.load(model_name+"_std.npy")
	autoencoder.restore_model()
	for j in range(0,npdata.shape[0],batch_size):
		inp_data = npdata[j:min(j+batch_size,npdata.shape[0])]
		data = autoencoder.get_reconstruction(inp_data)
		data = sbh.denormalize(data,mean,std)
		ret = []
		for i in range(len(data)):
			ret.append(sbh.convert_to_absolute(data[i]))
			with open("output/"+str(j).zfill(5)+".bvh",'w') as ofd:
				writer = csv.writer(ofd,delimiter=" ")
				for k in range(len(ret[i])):
					writer.writerow(ret[i][k])
			break






