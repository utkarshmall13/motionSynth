import numpy as np
import numpy.linalg as npla

def readBVH(fname):
	return np.loadtxt(fname,delimiter=' ')

def represent20(frames):
	frames = frames.reshape((-1,37,3))
	frames = frames[:,[0,1,2,3,4,7,8,9,10,13,14,16,17,18,20,21,22,29,30,31],:]
	frames = frames.reshape((-1,60))
	return frames

def getOrientation(frames):
	frames = frames.reshape((-1,20,3))
	triangle = frames[:,[0,14,17],:]
	triangle = triangle - triangle[:,[0],:]
	cross = np.cross(triangle[:,1,:],triangle[:,2,:])
	cross = cross/np.array([npla.norm(cross,axis = 1)]).T
	cross[:,1] = 0
	Z = cross/np.array([npla.norm(cross,axis = 1)]).T
	Y = np.array([0,1,0])
	X = np.cross(Y,Z)
	frames = frames.reshape((-1,60))
	frames = np.concatenate((frames,X,Z),axis=1)
	return frames

def subsample(frames,diff=4):
	ret = []
	for i in range(diff):
		ret.append(frames[i::diff])
	return ret

def convert_to_relative(frames):
	dir_facing = frames[:,63:66]
	# print(dir_facing)
	angle = np.rad2deg(np.arccos(np.einsum("ij,ij->i",dir_facing[:-1,:],dir_facing[1:,:])))
	dir = np.sign(np.cross(dir_facing[:-1,:],dir_facing[1:,:])[:,1])
	angle = angle*dir
	velocity = frames[1:,:3]-frames[:-1,:3]
	vz = np.einsum("ij,ij->i",velocity,dir_facing[:-1,:])
	vx = np.einsum("ij,ij->i",velocity,frames[:-1,60:63])

	temp = np.copy(frames[:-1,:3])
	temp[:,1] = 0
	temp = np.tile(temp,(1,19))

	frames[:-1,3:60] = frames[:-1,3:60]-temp
	ret = frames[:-1,:61]
	ret[:,0] = vx
	ret[:,2] = vz
	ret[:,-1] = angle
	return ret


def find_mean_std(framess):
	whole_data = np.concatenate(framess,axis = 0)
	return np.mean(whole_data,axis=0),np.std(whole_data,axis=0)

def normalize(framess,mean,std):
	ret = []
	for frames in framess:
		frames = (frames-mean)/std
		ret.append(frames)
	return ret

# # Test
# frames = readBVH('../../dataset/spatial/08/08_01.bvh')
# frames = represent20(frames)
# frames = getOrientation(frames)
# frames = subsample(frames)
# for i in range(4):
# 	frames[i] = convert_to_relative(frames[i])

# mean,std = find_mean_std(frames)
# print(normalize(frames,mean,std))