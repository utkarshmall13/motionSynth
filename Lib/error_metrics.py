import matplotlib.pyplot as plt
from bvhHandler import readBVH
import numpy as np
import math

#calculate error metrics
def calculateErrorMetricsSumFrame(file1,warmup1,file2,warmup2):
	data1 = readBVH(file1)
	data2 = readBVH(file2)
	if(len(data1)-2*warmup1!=len(data2)-2*warmup2):
		print(len(data1))
		print(len(data2))
		print("Incorrect offset or file out of sync")
		return ([],[])
	mat = []
	for i in range(warmup1,len(data1)-warmup1):
		vec = []
		for j in range(len(data1[i])):
			vec.append(data1[i][j]-data2[i+warmup2-warmup1][j])
		mat.append(vec)
	L1 = [0.0]*len(mat[0])
	L2 = [0.0]*len(mat[0])
	for i in range(len(mat)):
		for j in range(len(mat[i])):
			L1[j]+=math.fabs(mat[i][j])
			L2[j]+=(mat[i][j]*mat[i][j])
	for i in range(len(L1)):
		L1[i] = L1[i]/(len(mat))
		L2[i] = L2[i]/(len(mat))
	return (L1,L2)

def calculateErrorMetricsSingleFrame(file1,warmup1,file2,warmup2,frame):
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
	L1 = [0.0]*len(mat[0])
	L2 = [0.0]*len(mat[0])
	for j in range(len(mat[frame])):
		L1[j]=math.fabs(mat[frame][j])
		L2[j]=mat[i][j]*mat[frame][j]
	return (L1,L2)

def calculateErrorMetricsSumChannel(file1,warmup1,file2,warmup2):
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
	L1 = [0.0]*len(mat)
	L2 = [0.0]*len(mat)
	for i in range(len(mat)):
		for j in range(6,len(mat[i])):
			L1[i]+=math.fabs(mat[i][j])
			L2[i]+=(mat[i][j]*mat[i][j])
	for i in range(len(L1)):
		L1[i] = L1[i]/(len(mat[0])-6)
		L2[i] = L2[i]/(len(mat[0])-6)
	return (L1,L2)

def calculateErrorMetricsSingleChannel(file1,warmup1,file2,warmup2,channel):
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
	L1 = [0.0]*len(mat)
	L2 = [0.0]*len(mat)
	for i in range(len(mat)):
		L1[i]=math.fabs(mat[i][channel])
		L2[i]=mat[i][j]*mat[i][channel]
	return (L1,L2)

def calculateErrorMetrics(file1,warmup1,file2,warmup2):
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
	L1 = 0.0
	L2 = 0.0
	for i in range(len(mat)):
		for j in range(len(mat[i])):
			L1+=math.fabs(mat[i][j])
			L2+=(mat[i][j]*mat[i][j])
	L1 = L1/(len(mat)*len(mat[0]))
	L2 = L2/(len(mat)*len(mat[0]))
	return (L1,L2)


def plotErrors(files,offsets,legends,title,xlabel,ylabel,basefile,basefileOffset,L1orL2,metric,channel,frame,output_fig):
	Data = []
	for i in range(len(files)):
		print(files[i])
		if(metric=='SingleChannel'):
			(L1,L2) = calculateErrorMetricsSingleChannel(files[i],basefileOffset,basefile,offsets[i],channel)
			Data.append([L1,L2])
		elif(metric=='SumChannel'):
			(L1,L2) = calculateErrorMetricsSumChannel(files[i],basefileOffset,basefile,offsets[i])
			Data.append([L1,L2])
		elif(metric=='SingleFrame'):
			(L1,L2) = calculateErrorMetricsSingleFrame(files[i],basefileOffset,basefile,offsets[i],frame)
			Data.append([L1,L2])
		elif(metric=='SumFrame'):
			(L1,L2) = calculateErrorMetricsSumFrame(files[i],basefileOffset,basefile,offsets[i])
			Data.append([L1,L2])
		else:
			print("Metric not found")
			return
	for i in range(len(Data)):
		if(L1orL2=='L1'):
			Data[i] = Data[i][0]
		elif(L1orL2=='L2'):
			Data[i] = Data[i][1]
		else:
			print("Loss not found")
			return
	#print(Data)
	plt.figure(figsize=(15, 7))
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	for i in range(len(Data)):
		if(metric=='SumFrame' or metric=='SingleFrame'):
			gap =  0
		else:
			gap = offsets[i]
		plt.plot(list(range(gap,len(Data[i])+gap)),Data[i],label = legends[i],linewidth=2)
	plt.legend()
	plt.savefig(output_fig)
