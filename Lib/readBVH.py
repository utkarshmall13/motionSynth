from io import StringIO
import helper
import numpy as np

def readFiles(folder, files):
	data=[]
	i=1
	print("Reading...")
	for fileName in files:
		#directory for input data
		pr = int(30*i/len(files))
		progress = ' ['+'='*pr+" "*(30-pr)+']'
		print(str(i)+"/"+str(len(files))+": "+fileName+progress, end='\r')
		if(i==len(files)):
			print()
		text = open(folder+fileName+'.bvh').read()
		index = text.find('MOTION')
		text = text[index:]
		index = text.replace('\n', 'X', 2).find('\n')
		inner_data = []
		with StringIO(text[index+1:]) as data_file:
			for line in data_file:
				tmp=line.strip().split(' ')
				inner_data.append([float(temp) for temp in tmp])
			#inner_data=inner_data[warmUp:]
		data.append(inner_data[:])
		i = i + 1
	print()
	return data

def readFilesX(folder, files, extension, augment):
	data=[]
	i=1
	print("Reading...")
	for fileName in files:
		#directory for input data
		pr = int(30*i/len(files))
		progress = ' ['+'='*pr+" "*(30-pr)+']'
		print(str(i)+"/"+str(len(files))+": "+fileName+progress, end='\r')
		if(i==len(files)):
			print()
		for j in range(augment):
			text = open(folder+fileName+extension+'.bvh').read()
			index = text.find('MOTION')
			text = text[index:]
			index = text.replace('\n', 'X', 2).find('\n')
			inner_data = []
			with StringIO(text[index+1:]) as data_file:
				for line in data_file:
					tmp=line.strip().split(' ')
					inner_data.append([float(temp) for temp in tmp])
				#inner_data=inner_data[warmUp:]
			data.append(inner_data[:])
		i = i + 1
	print()
	return data

def readFilesNoisyX(folder, files, extension, augment):
	data=[]
	i=1
	print("Reading...")
	for fileName in files:
		#directory for input data
		pr = int(30*i/len(files))
		progress = ' ['+'='*pr+" "*(30-pr)+']'
		print(str(i)+"/"+str(len(files))+": "+fileName+progress, end='\r')
		if(i==len(files)):
			print()
		for j in range(augment):
			text = open(folder+fileName+extension+'_'+str(j+1)+'.bvh').read()
			index = text.find('MOTION')
			text = text[index:]
			index = text.replace('\n', 'X', 2).find('\n')
			inner_data = []
			with StringIO(text[index+1:]) as data_file:
				for line in data_file:
					tmp=line.strip().split(' ')
					inner_data.append([float(temp) for temp in tmp])
				#inner_data=inner_data[warmUp:]
			data.append(inner_data[:])
		i = i + 1
	print()
	return data


def remove_Tpose(data, warmUp):
	for i in range(len(data)):
		data[i] =  data[i][warmUp:]
	return data

def remove_HeadTail(data, head, tail):
	for i in range(len(data)):
		data[i] =  data[i][head:-tail]
	return data

def switch_rootrotn2_YZX(data):
	for singleFile in data:
		for frame in singleFile:
			z,y,x = helper.ZYX2YXZ(frame[3],frame[4],frame[5])
			frame[4],frame[5],frame[3] = z,y,x
	return data

def switch_rootrotn2_ZYX(data):
	for singleFile in data:
		for frame in singleFile:
			z,y,x = helper.YXZ2ZYX(frame[4],frame[5],frame[3])
			frame[3],frame[4],frame[5] = z,y,x
	return data

def mean(data):
    npdata=[]
    for inner_data in data:
    	for frame in inner_data:
    		npdata.append(frame)
    npdata = np.array(npdata)
    chanMean=npdata.mean(axis=0)
    return chanMean

def std(data):
    npdata=[]
    for inner_data in data:
    	for frame in inner_data:
    		npdata.append(frame)
    npdata = np.array(npdata)
    chanStD=npdata.std(axis=0)
    for i in range(len(chanStD)):
    	if(chanStD[i]<0.01):
    		chanStD[i]=0.01
    return chanStD
