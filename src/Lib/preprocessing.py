import numpy as np

def add_noise(XList, noise):
	X = np.array(XList)
	s = X.shape
	X = X.reshape(-1, s[-1])
	X = np.random.normal(X, noise)
	X = X.reshape(s)
	return X.tolist()

def repeat(dataList, augment):
	for i in range(len(dataList)):
		dataList[i] = dataList[i]*augment
	return dataList

def append_to_batchsize(dataList, batch_size):
	extra = batch_size - len(dataList[0])%batch_size
	rand = np.random.randint(0,len(dataList[0])-1, extra)
	for i in range(extra):
		for i in range(len(dataList)):
			dataList[i].append(dataList[i][rand[i]])
	return dataList

def shuffle(dataList):
	rng_state = np.random.get_state()
	for i in range(len(dataList)):
		np.random.set_state(rng_state)
		np.random.shuffle(dataList[i])
	return dataList

#returns Z and X of data for each file separately
def get_ZX_BNN_Test(data,diff,n_steps):
	XList = []
	ZList = []
	for index in range(len(data)):
		Xlist = []
		Zlist = []
		for i in range(diff, len(data[index])-n_steps*diff):
			XTemp=[]
			for j in range(n_steps):
				frame=[]
				frame = frame + [data[index][i+j*diff][3] - data[index][i+j*diff-diff][3]]
				frame = frame + [data[index][i+j*diff][4] - data[index][i+j*diff-diff][4]]
				frame = frame + [data[index][i+j*diff][5] - data[index][i+j*diff-diff][5]]
				frame = frame + data[index][i+j*diff][6:132]
				XTemp.append(frame)
			Xlist.append(XTemp)
			Zlist.append(data[index][i+int(n_steps/2)*diff][0:6])
		XList.append(Xlist)
		ZList.append(Zlist)
	return (ZList,XList)

#returns Z and X of data for each file together
def get_ZX_BNN(data,diff,n_steps):
	XList = []
	ZList = []
	for index in range(len(data)):
		for i in range(diff, len(data[index])-n_steps*diff):
			XTemp=[]
			for j in range(n_steps):
				frame=[]
				frame = frame + [data[index][i+j*diff][3] - data[index][i+j*diff-diff][3]]
				frame = frame + [data[index][i+j*diff][4] - data[index][i+j*diff-diff][4]]
				frame = frame + [data[index][i+j*diff][5] - data[index][i+j*diff-diff][5]]
				frame = frame + data[index][i+j*diff][6:132]
				XTemp.append(frame)
			XList.append(XTemp)
			ZList.append(data[index][i+int(n_steps/2)*diff][0:6])
	return (ZList,XList)
