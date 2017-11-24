from __future__ import print_function
import numpy as np
import random
import sys
from os import walk
from io import StringIO
import math
import csv
import helper
import bvh23d

#TODO write sanity check for ssize to be power of two
def getFraction(val, ssize, index,gsamples):
	#if(ssize>16):
		#ssize = 16
	if(index<=-ssize or index>=ssize):
		return 0
	return val*(gsamples[ssize][ssize-1+index])

########################################################################
#function creates bvh, takes 2d list creates bvh for cmu data
def createBVH(Result,fps,output_file,BVHHeader):
		with open (BVHHeader, "r") as BVHHFile:
			header=BVHHFile.readlines()
		with open (output_file, "w") as outputFile:
			outputFile.writelines(header)
			outputFile.write('MOTION\n')
			outputFile.write('Frames: '+str(len(Result))+'\n')
			outputFile.write('Frame Time: '+str(1.0/fps)+'\n')
			for i in range(len(Result)):
				line=''
				for x in range(len(Result[0])):
					line+=str(float(Result[i][x]))+' '
				line+='\n'
				outputFile.write(line)

#function reads bvh return 2d list
def readBVH(fileName):
	text = open(fileName).read()
	index = text.find('MOTION')
	text = text[index:]
	index = text.replace('\n', 'X', 2).find('\n')
	frames = []
	with StringIO(text[index+1:]) as data_file:
		i=0
		for line in data_file:
			tmp=line.strip().split(' ')
			frames.append([float(temp) for temp in tmp])
	return frames

#changes the order of rotation INPLACE return void
#i.e. y is the first in order of rotation, even though y is still at index 4
def swapRotationOrder(frames):
	for tmp in frames:
		z,y,x = helper.ZYX2YXZ(tmp[3],tmp[4],tmp[5])
		tmp[4],tmp[5],tmp[3] = z,y,x
#inverse of swapRotationOrder
def swapBackRotationOrder(frames):
	for tmp in frames:
		z,y,x = helper.YXZ2ZYX(tmp[4],tmp[5],tmp[3])
		tmp[3],tmp[4],tmp[5] = z,y,x

#swaps yrot and ytrans channel INPLACE return void
def swapYRotTrans(frames):
	for tmp in frames:
		swp = tmp[1]
		tmp[1] = tmp[4]
		tmp[4] = swp

#ignore first 3 channel, return first 3 channels separately and last channels separately
def ignoreGlobal(frames):
	ret = []
	frameRet = []
	for tmp in frames:
		ret.append(tmp[0:3])
		frameRet.append(tmp[3:])
	return (ret,frameRet)

def getCleaningBatches(X,Y,n_steps,Diff,batch_size):
	XList = []
	YList = []
	for i in range(len(X)-((n_steps-1)*Diff+1)+1):
		xdata = []
		ydata = []
		for j in range(n_steps):
			xsubdata = []
			for k in range(len(X[i+j*Diff])):
				xsubdata.append(X[i+j*Diff][k])
			xdata.append(xsubdata)
		for k in range(len(Y[i+int(n_steps/2)*Diff])):
			ydata.append(Y[i+int(n_steps/2)*Diff][k])
		YList.append(ydata)
		XList.append(xdata)
	while(len(XList)%batch_size!=0):
		rand = random.randint(0,len(XList)-1)
		XList.append(XList[rand][:])
		YList.append(YList[rand][:])
	ret_x =[]
	ret_y =[]
	#print("length of XLISt is:",len(XList))
	for i in range(0,len(XList),batch_size):
		ret_x.append(np.array(XList[i:i+batch_size]))
		ret_y.append(np.array(YList[i:i+batch_size]))
	return (ret_x,ret_y)

def getCleaningBatchesFromMultipleFiles(X,Y,n_steps,Diff,batch_size):
	if(len(X)!=len(Y)):
		print("ERROR len(X)!=len(Y)")
		return ([],[])
	XList = []
	YList = []
	for f in range(len(X)):
		for i in range(len(X[f])-((n_steps-1)*Diff+1)+1):
			xdata = []
			ydata = []
			for j in range(n_steps):
				xsubdata = []
				for k in range(len(X[f][i+j*Diff])):
					xsubdata.append(X[f][i+j*Diff][k])
				xdata.append(xsubdata)
			XList.append(xdata)
			for k in range(len(Y[f][i+int(n_steps/2)*Diff])):
				ydata.append(Y[f][i+int(n_steps/2)*Diff][k])
			YList.append(ydata)
		print("done: "+str(int(f)+1)+"/"+str(int(len(X))))
	while(len(XList)%batch_size!=0):
		rand = random.randint(0,len(XList)-1)
		XList.append(XList[rand][:])
		YList.append(YList[rand][:])
	ret_x =[]
	ret_y =[]
	#print("length of XLISt is:",len(XList))
	for i in range(0,len(XList),batch_size):
		ret_x.append(np.array(XList[i:i+batch_size]))
		ret_y.append(np.array(YList[i:i+batch_size]))
	print(str(len(ret_x))+" batches created")
	return (ret_x,ret_y)

#TODO: make more generic
#very very very specific, not even close to generic. not to be used until average subsampling at exactly 4 levels
def getCleaningBatchesFromMultipleFilesMultipleLevel(Xs,Y,n_steps,Diffs,batch_size):
	Diff = Diffs[-1]
	XList = []
	YList = []
	ret_x = []
	ret_y = []
	for X in Xs:
		XList.append([])
		ret_x.append([])
		if(len(X)!=len(Y)):
			print("ERROR len(X)!=len(Y)")
			return (XList,YList)

	for f in range(len(Y)):
		for i in range(Diff*int(n_steps/2),len(Y[f])-(Diff*int(n_steps/2))):
			for s in range(len(Diffs)):
				xdata = []
				for j in range(-int(n_steps/2),int(n_steps/2)+1):
					xsubdata = []
					for k in range(len(Xs[s][f][i+j*Diffs[s]])):
						xsubdata.append(Xs[s][f][i+j*Diffs[s]][k])
					xdata.append(xsubdata)
				XList[s].append(xdata)
			ydata = []
			for k in range(len(Y[f][i])):
				ydata.append(Y[f][i][k])
			YList.append(ydata)
		print("done: "+str(int(f)+1)+"/"+str(int(len(Y))))
	while(len(YList)%batch_size!=0):
		rand = random.randint(0,len(XList)-1)
		for s in range(len(Diffs)):
			XList[s].append(XList[s][rand][:])
		YList.append(YList[rand][:])
	#print("length of XLISt is:",len(XList))
	for i in range(0,len(YList),batch_size):
		for s in range(len(Diffs)):
			ret_x[s].append(np.array(XList[s][i:i+batch_size]))
		ret_y.append(np.array(YList[i:i+batch_size]))
	print(str(len(ret_y))+" batches created")
	return (ret_x,ret_y)

def arrangeTestFile(fileName,Diffs,noise,gt_dir,noisy_dir):
	Y=[]
	X=[]
	for diff in Diffs:
		X.append([])
	yy = readBVH(gt_dir+fileName+'_s1'+'.bvh')
	swapRotationOrder(yy)
	swapYRotTrans(yy)
	(global_motion,yy) = ignoreGlobal(yy)
	Y.append(yy)
	for s in range(len(Diffs)):
		Diff = Diffs[s]
		xx = readBVH(noisy_dir+fileName+'_s'+str(Diff)+'_n'+str(noise)+'_1'+'.bvh')
		swapRotationOrder(xx)
		swapYRotTrans(xx)
		(garb,xx) = ignoreGlobal(xx)
		X[s].append(xx)
	return (X,Y,global_motion)

########################################################################
########################################################################
########################################################################
########################################################################

#If the first 3 values in each frame is xTrans,yTrans,zTrans,xRot,yRot,zRot
#converts it to xVel,yTrans,zVel,xRot,yAngVel,zRot INPLACE
#velFrameDiff is the difference between frames on which to calculate difference
def getVelocities(frames,velFrameDiff):
	vel=[]
	index=0
	for i in range(0,6,2):
		vel.append([])
		for j in range(len(frames)-velFrameDiff):
			vel[index].append((frames[j+velFrameDiff][i]-frames[j][i]))
		for j in range(velFrameDiff):
			vel[index].append(frames[j-velFrameDiff][i]-frames[j-2*velFrameDiff][i])
		index+=1
	velComp=[[],[]]
	for i in range(len(vel[0])):
		velComp[0].append(math.sin(math.radians(frames[i][4]))*vel[0][i]+math.cos(math.radians(frames[i][4]))*vel[1][i])
		velComp[1].append(math.cos(math.radians(frames[i][4]))*vel[0][i]-math.sin(math.radians(frames[i][4]))*vel[1][i])
	for i in range(0,4,2):
		for j in range(len(frames)):
			frames[j][i]=velComp[int(i/2)][j]

	for j in range(len(frames)):
		frames[j][4]=vel[2][j]

def smoothData(frames,smoothingWindow,warm_up,gsamples,channel_count):
	#initialize a zer0 2d list
	mat = []
	for i in range(len(frames)):
		vec = [0.0]*channel_count
		mat.append(vec)
	#add gaussian multiplied fraction
	for i in range(len(frames)):
		for k in range(i-smoothingWindow+1,i+smoothingWindow):
			if(k<0 or k>=len(frames)):
				continue
			for j in range(channel_count):
				mat[k][j] += getFraction(frames[i][j],smoothingWindow,k-i,gsamples)
	return mat[warm_up:len(mat)-warm_up]


########################################################################
########################################################################
#duh!
def calculateChanMeanStd(files,channel_count):
	meanCount=0.0
	chanMean = [0.0]*channel_count
	chanStD = [0.0]*channel_count

	for sfile in files:
		singleFile = readBVH(sfile)
		swapRotationOrder(singleFile)
		for frame in singleFile:
			meanCount+=1
			for i in range(len(frame)):
				chanMean[i]+=frame[i]

	for i in range(channel_count):
		chanMean[i]/=meanCount

	for sfile in files:
		singleFile = readBVH(sfile)
		swapRotationOrder(singleFile)
		for frame in singleFile:
			for i in range(len(frame)):
				chanStD[i]+=((frame[i]-chanMean[i])*(frame[i]-chanMean[i]))

	for i in range(channel_count):
		chanStD[i]/=meanCount
		chanStD[i]=math.sqrt(chanStD[i])
		if(chanStD[i]<0.01):
			chanStD[i]=0.01
	chanStD[1]=0.1
	return(chanMean,chanStD)

#adds noise to last 129 channels + x and z rot, takes 2d list returns 2d list
def addNoise(frames,chanStD,channel_count,noise):
	mat = []
	for frame in frames:
		vec = []
		for i in range(6):
			'''if(i%2==0):
				vec.append(frame[i])
			else:
				vec.append(np.random.normal(frame[i],chanStD[i]*noise,1)[0])'''
			vec.append(frame[i])
		for i in range(6,channel_count):
			vec.append(np.random.normal(frame[i],chanStD[i]*noise,1)[0])
		mat.append(vec)
	return mat

def addSpatialNoise(frames,noise,BVHHeader):
	header = bvh23d.readHeader(BVHHeader)
	tree = bvh23d.Tree(header)

	frames_new = []

	for i in range(0,len(frames)):
		tree.set_dof_values(frames[i])
		tree.set_transformations_values()
		tree.add_noise(noise)
		tree.set_corrected_position()
		tree.set_corrected_dofs(noise)
		frames_new.append(tree.get_corrected_dofs())
	return frames_new

def addSpatialNoiseSomeChannel(frames,noise,BVHHeader):
	header = bvh23d.readHeader(BVHHeader)
	tree = bvh23d.Tree(header)

	frames_new = []

	for i in range(0,len(frames)):
		tree.set_dof_values(frames[i])
		tree.set_transformations_values()
		tree.add_noise_sc(noise)
		tree.set_corrected_position()
		tree.set_corrected_dofs(noise)
		frames_new.append(tree.get_corrected_dofs())
	return frames_new

def addSpatialNoiseVary(frames,noise,BVHHeader):
	header = bvh23d.readHeader(BVHHeader)
	tree = bvh23d.Tree(header)

	frames_new = []

	for i in range(0,len(frames)):
		tree.set_dof_values(frames[i])
		tree.set_transformations_values()
		tree.add_noise_vary(noise)
		tree.set_corrected_position()
		tree.set_corrected_dofs(noise)
		frames_new.append(tree.get_corrected_dofs())
	return frames_new

def addBiasedSineNoise(frames,chanStD,channel_count,noise):
	mat = []
	L  = len(frames)
	bias = 7*np.sin(np.arange(L)*np.pi/10)
	for index in range(L):
		frame = frames[index]
		vec = []
		for i in range(6):
			vec.append(frame[i])
		for i in range(6,channel_count):
			vec.append(np.random.normal(frame[i] + bias[index],chanStD[i]*noise,1)[0])
		mat.append(vec)
	return mat

def addBiasedNoise(frames,chanStD,channel_count,noise):
	mat = []
	L  = len(frames)
	bias = 7*(np.arange(L)*0+1)
	for index in range(L):
		frame = frames[index]
		vec = []
		for i in range(6):
			vec.append(frame[i])
		for i in range(6,channel_count):
			vec.append(np.random.normal(frame[i] + bias[index],chanStD[i]*noise,1)[0])
		mat.append(vec)
	return mat


def addUniformNoise(frames,chanStD,channel_count,noise):
	mat = []
	for frame in frames:
		vec = []
		for i in range(6):
			vec.append(frame[i])
		for i in range(6,channel_count):
			vec.append(np.random.uniform(frame[i]-3*chanStD[i]*noise,frame[i]+3*chanStD[i]*noise,1)[0])
		mat.append(vec)
	return mat

#adds noise to every channel, takes 2d list returns 2d list
def addFullNoise(frames,chanStD,channel_count,noise):
	mat = []
	for frame in frames:
		vec = []
		for i in range(channel_count):
			vec.append(np.random.normal(frame[i],chanStD[i]*noise,1)[0])
		mat.append(vec)
	return mat


#adds gap to last 126 channels takes 2d list returns 2d list
def addGap(frames,channel_count,minlen,maxlen,prob):
	ind =[list(range(6, 17)),list(range(27, 35)),list(range(45, 47)),list(range(51, 53)),list(range(57, 59)),
		list(range(63, 65)),list(range(69, 77)),list(range(87, 89)),list(range(93, 95)),list(range(99, 101)),
		list(range(105, 108)),list(range(111, 119)),list(range(123, 131))
		]
	indices = [item for sublist in ind for item in sublist]
	mat = []
	for frame in frames:
		vec = []
		for i in range(channel_count):
			vec.append(frame[i])
		mat.append(vec)
	for i in range(len(frames)):
		if(random.uniform(0,1)<prob):
			chan = indices[random.randint(0,len(indices)-1)]
			Range = random.randint(minlen,maxlen)
			for j in range(Range):
				if(i+j>=len(mat)):
					break
				mat[i+j][chan] = 0
	return mat


#adds gap to last 126 channels takes 2d list returns 2d list
def addGapNaN(frames,channel_count,minlen,maxlen,prob):
	ind =[list(range(6, 17)),list(range(27, 35)),list(range(45, 47)),list(range(51, 53)),list(range(57, 59)),
		list(range(63, 65)),list(range(69, 77)),list(range(87, 89)),list(range(93, 95)),list(range(99, 101)),
		list(range(105, 108)),list(range(111, 119)),list(range(123, 131))
		]
	indices = [item for sublist in ind for item in sublist]
	mat = []
	for frame in frames:
		vec = []
		for i in range(channel_count):
			vec.append(frame[i])
		mat.append(vec)
	for i in range(len(frames)):
		if(random.uniform(0,1)<prob):
			chan = indices[random.randint(0,len(indices)-1)]
			Range = random.randint(minlen,maxlen)
			for j in range(Range):
				if(i+j>=len(mat)):
					break
				mat[i+j][chan] = math.nan
	return mat

#adds gap to last 126 channels takes 2d list returns 2d list
def addRealisticGapNaN(frames,channel_count,decay,prob):
	ind =[list(range(6, 17)),list(range(27, 35)),list(range(45, 47)),list(range(51, 53)),list(range(57, 59)),
		list(range(63, 65)),list(range(69, 77)),list(range(87, 89)),list(range(93, 95)),list(range(99, 101)),
		list(range(105, 108)),list(range(111, 119)),list(range(123, 131))
		]

	indices = [item for sublist in ind for item in sublist]
	mat = []
	for frame in frames:
		vec = []
		for i in range(channel_count):
			vec.append(frame[i])
		mat.append(vec)
	for i in range(len(frames)):
		if(random.uniform(0,1)<prob):
			P = np.random.uniform()
			Range = int(math.ceil(-np.log(1-P)/decay))

			chan = indices[random.randint(0,len(indices)-1)]
			if(math.isnan(mat[i][chan])):
				continue
			for j in range(Range):
				if(i+j>=len(mat)):
					break
				mat[i+j][chan] = math.nan
	return mat


#calculate error metrics
def calulateErrorMetrics(file1,warmup1,file2,warmup2):
	data1 = readBVH(file1)
	data2 = readBVH(file2)
	if(len(data1)-2*warmup1!=len(data2)-2*warmup2):
		return ([],[])
	mat = []
	for i in range(warmup1,len(data1)-warmup1):
		vec = []
		for j in range(len(data1[i])):
			vec.append(data1[i][j]-data2[i+warmup2-warmup1][j])
		mat.append(vec)
	L1 = [0.0]*len(data1[i])
	L2 = [0.0]*len(data1[i])
	for i in range(len(mat)):
		for j in range(len(mat[i])):
			L1[j]+=math.fabs(mat[i][j])
			L2[j]+=(mat[i][j]*mat[i][j])
	for i in range(len(L1)):
		L1[i] = L1[i]/(len(mat))
		L2[i] = L2[i]/(len(mat))
	return (L1,L2)
