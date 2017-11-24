import math
from copy import deepcopy as dc

def fillGapUsingZero(data):
	channelCount = len(data[0])
	ret = dc(data)
	for j in range(channelCount):
		for i in range(len(ret)):
			if(math.isnan(ret[i][j])):
				ret[i][j] = 0
	return ret

def fillGapUsingInterpolation(data):
	channelCount = len(data[0])
	ret = dc(data)
	for j in range(channelCount):
		for i in range(len(ret)):
			if(math.isnan(ret[i][j])):
				startindex = i
				endindex = i
				while(i<len(ret) and math.isnan(ret[i][j])):
					i+=1
					endindex = i
				startval = 0
				endval = 0
				if(startindex==0 and endindex==len(ret)):
					startval = 0
					endval = 0
				elif(startindex==0):
					startval = ret[endindex][j]
					endval = ret[endindex][j]
				elif(endindex==len(ret)):
					startval = ret[startindex-1][j]
					endval = ret[startindex-1][j]
				else:
					startval = ret[startindex-1][j]
					endval = ret[endindex][j]
				for k in range(startindex,endindex):
					ret[k][j] = ((endindex-k)*startval+(k-startindex)*endval)/(endindex-startindex)
	return ret