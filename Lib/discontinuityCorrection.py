from copy import deepcopy as dc

def correct(data):
	ret = dc(data)
	channelCount = len(data[0])
	for j in range(3,channelCount):
		supplement = 0
		for i in range(len(data)-1):
			if(data[i][j]-data[i+1][j]>180):
				supplement+=360
			if(data[i+1][j]-data[i][j]>180):
				supplement-=360
			ret[i+1][j]+=supplement
	data = ret
