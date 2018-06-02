import numpy as np
import numpy.linalg as npla

def readBVH(fname):
	'''
	Read BVH
	'''
	return np.loadtxt(fname,delimiter=' ')

def represent20(frames):
	'''
	Get 20 representative joints
	'''
	frames = frames.reshape((-1,37,3))
	frames = frames[:,[0,1,2,3,4,7,8,9,10,13,14,16,17,18,20,21,22,29,30,31],:]
	frames = frames.reshape((-1,60))
	return frames

def getOrientation(frames):
	'''
	gets velocity,orientation of individual frame from positions
	'''
	frames = frames.reshape((-1,20,3))
	#hip and two  shoulder joints
	triangle = frames[:,[0,14,17],:]
	triangle = triangle - triangle[:,[0],:]
	#cross between two vectors to get front direction
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
	'''
	Subsamples data
	'''
	ret = []
	for i in range(diff):
		ret.append(frames[i::diff])
	return ret

def convert_to_relative(frames):
	'''
	find velocity in forward and perpendicular directions, angle of turn
	convert points to frame of forward motion
	'''
	dir_facing = frames[:,63:66]
	# print(dir_facing)
	# dot product
	angle = np.rad2deg(np.arccos(np.minimum(1,np.einsum("ij,ij->i",dir_facing[:-1,:],dir_facing[1:,:]))))
	dir = np.sign(np.cross(dir_facing[:-1,:],dir_facing[1:,:])[:,1])
	angle = angle*dir
	# print(frames[:,:3])
	velocity = frames[1:,:3]-frames[:-1,:3]
	# print(velocity)
	vz = np.einsum("ij,ij->i",velocity,dir_facing[:-1,:])
	vx = np.einsum("ij,ij->i",velocity,frames[:-1,60:63])

	temp = np.copy(frames[:-1,:3])
	temp[:,1] = 0
	temp = np.tile(temp,(1,19))

	frames[:-1,3:60] = frames[:-1,3:60]-temp
	for i in range(3,60,3):
		col_x = frames[:,i]*frames[:,60]+frames[:,i+2]*frames[:,62]
		col_z = frames[:,i]*frames[:,63]+frames[:,i+2]*frames[:,65]
		frames[:,i] = col_x
		frames[:,i+2] = col_z
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

##############################################################
#backward conversion
def denormalize(framess,mean,std):
	ret = []
	for frames in framess:
		frames = (frames*std+mean)
		ret.append(frames)
	return ret

def convert_to_absolute(frames,seed=[0,0,0]):
	angle = frames[:,-1]
	vx = frames[:,0]
	vz = frames[:,2]

	path_x = [seed[0]]
	path_y = [seed[1]]
	path_z = [seed[2]]

	ret = []
	for i in range(frames.shape[0]):
		path_x.append(path_x[-1] + vz[i]*np.sin(np.deg2rad(path_y[-1])) + vx[i]*np.cos(np.deg2rad(path_y[-1])))
		path_z.append(path_z[-1] + vz[i]*np.cos(np.deg2rad(path_y[-1])) - vx[i]*np.sin(np.deg2rad(path_y[-1])))
		path_y.append(path_y[-1] + angle[i])

		ret.append([path_x[i],frames[i,1],path_z[i]])
		for j in range(3,60,3):
			ret[-1].append(path_x[i]+frames[i,j+2]*np.sin(np.deg2rad(path_y[i])) + frames[i,j]*np.cos(np.deg2rad(path_y[i])))
			ret[-1].append(frames[i,j+1])
			ret[-1].append(path_z[i]+frames[i,j+2]*np.cos(np.deg2rad(path_y[i])) - frames[i,j]*np.sin(np.deg2rad(path_y[i])))
	return ret

# # # Test
# # frames = readBVH('../../dataset/spatial/01/01_01.bvh')
# # frames = represent20(frames)
# # frames = getOrientation(frames)
# # frames = subsample(frames,diff=4)
# # before = np.copy(frames[0][:,2])
# # seed = [frames[0][0,0],0,frames[0][0,2]]
# # frames[0] = convert_to_relative(frames[0])
# # mean,std = find_mean_std([frames[0]])
# # frames2 = denormalize(normalize([frames[0]],mean,std),mean,std)
# # # import numpy.linalg as npla
# # # print(npla.norm(frames2[0]-frames[0]))
# # output = convert_to_absolute(frames2[0],seed = seed)

# print(output)
# import csv
# with open("out2.csv",'w') as ofd:
# 	writer = csv.writer(ofd,delimiter=" ")
# 	for i in range(len(output)):
# 		writer.writerow(output[i])
