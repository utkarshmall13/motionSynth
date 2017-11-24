import readBVH
import preprocessing
import writeBVH
import numpy as np
import csv

noise =0.5

files=['07_02', '08_02', '09_02', '16_12', '69_27', '69_31', '91_40', '74_04']

folder_pred = './CMU_data(Clean_'+str(noise)+')/autoencoder/'
folder_true = '../Data/gt/'

data_true = readBVH.readFilesX(folder_true, files, extension='_s1', augment=5)
data_pred = readBVH.readFilesNoisyX(folder_pred, files, extension='', augment=5)

print("Computing Mean Error Across all Channels and Frames...")
errList=[]
for i in range(len(data_pred)):
	XList= []
	YList= []

	for j in range(len(data_pred[i])):
		XList.append(data_pred[i][j][6:])
		YList.append(data_true[i][j][6:])
	X = np.array(XList)
	Y = np.array(YList)
	err = X-Y
	err = np.square(err)
	err = np.mean(err)
	errList.append([files[int(i/5)],err])

	pr = int(30*i/len(data_pred))
	progress = ' ['+'='*pr+" "*(30-pr)+']'
	print(str(i+1)+"/"+str(len(data_pred))+": "+files[int(i/5)]+progress, end='\r')
	if(i==len(files)-1):
		print()

with open(folder_pred+"error.csv", "w") as f:
#with open("error.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(errList)
