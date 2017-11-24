from io import StringIO
import numpy as np
from copy import deepcopy as dc
from numpy.random import normal as norm
from numpy.linalg import norm as norm2
from numpy.linalg import svd,det
from scipy.sparse.linalg import eigs as eig

import math

np.random.seed(42)

np.set_printoptions(precision=4,suppress=True)


class Node:
	def __init__(self,name,dofs,offset,isreal=True):
		self.name = name
		self.isreal = isreal
		self.dofs = dofs
		self.offset = offset
		self.children = []
		self.parent = None
		self.depth = 0

	def is_leaf(self):
		if(len(self.children)==0):
			return True
		return False

	def is_parent(self):
		if(self.parent == None):
			return True
		return False

	def add_child(self,node):
		node.parent = self
		self.children.append(node)


class Tree:
	def __init__(self,header):
		assert header[0]=='HIERARCHY'
		header = header[1:]
		self.root = self.create_tree(header)
		self.set_depth()

	def set_depth(self):
		self.set_depth_helper(self.root,0)
	def set_depth_helper(self,node,depth):
		node.depth = depth
		for child in node.children:
			self.set_depth_helper(child,depth+1)


	def create_tree(self, header):
		name = header[0]
		offset = self.get_offset(header[2])
		dofs = self.get_dofs(header[3])
		node = Node(name,dofs,offset)
		# print(header[0])
		if(header[0]!="ROOT hip"):
			offs = self.getPerps(offset)
			node2 = Node("fake af",[],offs[0],isreal=False)
			node3 = Node("fake af",[],offs[1],isreal=False)
		
		children = self.find_children(header[4:-1])
		for child in children:
			node.add_child(child)

		if(header[0]!="ROOT hip"):
			return [node,node2,node3]
		return node


	def get_offset(self,line):
		return [float(f) for f in line.split(' ')[1:]]

	def get_dofs(self,line):
		return [f for f in line.split(' ')[2:]]

	def children_lines(self,header):
		assert header[1]=='{'
		assert header[-1]=='}'
		ret = []
		counter = 0
		for i in range(len(header)):
			if(header[i]=='{'):
				if(counter==0):
					start = i-1
				counter+=1
			elif(header[i]=='}'):
				counter-=1
				if(counter==0):
					end = i+1
					ret.append((start,end))
		return ret

	def getPerps(self,offset):
		if(mod(offset[0])<0.0001 and mod(offset[1])<0.0001):
			v1 = [offset[2],0,0]
		elif(mod(offset[0])>mod(offset[1])):
			v1 = [-offset[1]/offset[0],1,0]
		else:
			v1 = [1,-offset[0]/offset[1],0]

		v0 = np.array(offset)
		normv0 = norm2(v0)
		v0= v0/normv0

		v1 = np.array(v1)
		normv1 = norm2(v1)
		v1 = v1/normv1

		v2 = np.cross(v0,v1)

		v0 = v0*normv0
		v1 = v1*normv0
		v2 = v2*normv0

		return [v1.tolist(),v2.tolist()]

	def find_children(self,header):
		#base case
		if(len(header)<=4):
			node = Node(header[0],[],self.get_offset(header[2]))
			offs = self.getPerps(self.get_offset(header[2]))
			node2 = Node("fake af",[],offs[0],isreal=False)
			node3 = Node("fake af",[],offs[1],isreal=False)

			return [node,node2,node3]
		indices = self.children_lines(header)
		nodes = []
		for ind in indices:
			adding_nodes = self.create_tree(header[ind[0]:ind[1]])
			for node in adding_nodes:
				# print(node.name)
				nodes.append(node)
		return nodes

	def traverse_tree(self):
		self.traverse_tree_helper(self.root,0)

	def traverse_tree_helper(self,node,offset):
		print("\t"*offset, node.name+' '+str(node.depth))
		for child in node.children:
			self.traverse_tree_helper(child,offset+1)

	def get_corrected_dofs(self):
		return self.get_corrected_dofs_helper(self.root)

	def get_corrected_dofs_helper(self,node):
		frame = []
		frame+=node.corrected_dof_values
		for child in node.children:
			if(child.isreal):
				frame+=self.get_corrected_dofs_helper(child)
		return frame



	def set_dof_values(self,frame):
		self.total_dofs = self.set_dof_values_helper(self.root,frame,0)

	def set_dof_values_helper(self,node,frame,count):
		if(node.dofs==[]):
			node.dof_values = []
			return count
		node.dof_values = frame[count:count+len(node.dofs)]
		counter = count+len(node.dofs)
		for child in node.children:
			counter = self.set_dof_values_helper(child,frame,counter)
		return counter

	def get_position(self):
		return self.get_position_helper(self.root)
	def get_position_helper(self,node):
		ret = [(node.position[0],node.position[1],node.position[2])]
		for child in node.children:
			if("fake" not in child.name):
				ret+=self.get_position_helper(child)
		return ret



	def set_transformations_values(self):
		transform = np.identity(4)
		self.set_transformations_values_helper(self.root,transform)

	def set_transformations_values_helper(self,node,transform):
		transform_new = dc(transform)
		node.transform = transform_new
		node.position = vec4tovec3(node.transform.dot(vec3tovec4(node.offset)))
		transform_new = node.transform.dot(translation(node.offset))

		for i in range(len(node.dofs)):
			dof = node.dofs[i]
			dof_val = node.dof_values[i]
			if(dof=="Xposition"):
				trans = translation((dof_val,0,0))
				transform_new = transform_new.dot(trans)
			elif(dof=="Yposition"):
				trans = translation((0,dof_val,0))
				transform_new = transform_new.dot(trans)
			elif(dof=="Zposition"):
				trans = translation((0,0,dof_val))
				transform_new = transform_new.dot(trans)
			elif(dof=="Xrotation"):
				rot = rotation_x(dof_val)
				transform_new = transform_new.dot(rot)
			elif(dof=="Yrotation"):
				rot = rotation_y(dof_val)
				transform_new = transform_new.dot(rot)
			elif(dof=="Zrotation"):
				rot = rotation_z(dof_val)
				transform_new = transform_new.dot(rot)
		for child in node.children:
			self.set_transformations_values_helper(child,dc(transform_new))

	def print_pcloud(self):
		self.print_pcloud_helper(self.root)

	def print_pcloud_helper(self,node):
		print(node.corrected_position[0],node.corrected_position[1],node.corrected_position[2],0,0,1)
		for child in node.children:
			self.print_pcloud_helper(child)

	def add_noise(self,noise_std):
		self.add_noise_helper(self.root,noise_std)

	def add_noise_helper(self,node,noise_std):
		if(not (node.depth==0 or node.depth==1)):
			node.noised_position = [norm(node.position[0],noise_std),norm(node.position[1],noise_std),norm(node.position[2],noise_std)]
		else:
			node.noised_position = node.position[:3]
		for child in node.children:
			self.add_noise_helper(child,noise_std)

	def add_noise_sc(self,noise_std):
		self.add_noise_sc_helper(self.root,noise_std)

	def add_noise_sc_helper(self,node,noise_std):
		if(not (node.depth==0 or node.depth==1)):
			if("Hand" in node.name or "Foot" in node.name):
				node.noised_position = [norm(node.position[0],noise_std),norm(node.position[1],noise_std),norm(node.position[2],noise_std)]
			else:
				node.noised_position = node.position[:3]
		else:
			node.noised_position = node.position[:3]
		for child in node.children:
			self.add_noise_sc_helper(child,noise_std)

	def add_noise_vary(self,noise_std):
		self.add_noise_vary_helper(self.root,noise_std)

	def add_noise_vary_helper(self,node,noise_std):
		if(not (node.depth==0 or node.depth==1)):
			if("Hand" in node.name or "Foot" in node.name):
				node.noised_position = [norm(node.position[0],noise_std),norm(node.position[1],noise_std),norm(node.position[2],noise_std)]
			else:
				node.noised_position = [norm(node.position[0],0.1),norm(node.position[1],0.1),norm(node.position[2],0.1)]
		else:
			node.noised_position = node.position[:3]
		for child in node.children:
			self.add_noise_vary_helper(child,noise_std)

	def set_corrected_position(self):
		self.set_corrected_position_helper(self.root)
	def set_corrected_position_helper(self,node):
		# if(not (node.name=="ROOT hip" or node.name=="JOINT abdomen")):
		if(not (node.depth==0 or node.depth==1)):
			diff = np.array(node.noised_position)-np.array(node.parent.corrected_position)
			lengthratio = norm2(diff)/norm2(node.offset)
			diff = diff/lengthratio
			diff = diff+np.array(node.parent.corrected_position)
			node.corrected_position = diff.tolist()
		else:
			node.corrected_position = node.position[:3]
		for child in node.children:
			self.set_corrected_position_helper(child)

	def set_corrected_dofs(self,noise_std):
		transform = np.identity(4)
		self.set_corrected_dofs_helper(self.root,transform)

	def set_corrected_dofs_helper(self,node,transform):
		transform_new = dc(transform)
		node.corrected_transform = transform_new
		# node.corrected_position = node.corrected_transform.dot(vec3tovec4(node.offset))
		transform_new = dc(node.corrected_transform.dot(translation(node.offset)))
		if(node.name=="End Site" or not node.isreal):
			node.corrected_dof_values = []
			return
		if(node.depth==0):
		# if(True):
			inter_transform = np.identity(4)
			for i in range(len(node.dofs)):
				dof = node.dofs[i]
				dof_val = node.dof_values[i]
				rot = np.identity(4)
				if(dof=="Xposition"):
					rot = translation((dof_val,0,0))
				elif(dof=="Yposition"):
					rot = translation((0,dof_val,0))
				elif(dof=="Zposition"):
					rot = translation((0,0,dof_val))
				elif(dof=="Xrotation"):
					rot = rotation_x(dof_val)
				elif(dof=="Yrotation"):
					rot = rotation_y(dof_val)
				elif(dof=="Zrotation"):
					rot = rotation_z(dof_val)
				inter_transform = inter_transform.dot(rot)
				# transform_new = transform_new.dot(rot)

			transform_new = transform_new.dot(inter_transform)
			node.corrected_dof_values = node.dof_values[:]
			node.noised_position = node.position[:3]
			node.corrected_position = node.noised_position[:]
			for child in node.children:
				child.noised_position = child.position[:3]
				child.corrected_position = child.noised_position[:]
		else:
			TA = np.empty((len(node.children),3))
			i=0
			finapos = []
			initpos = []

			for child in node.children:
				vec = vec4tovec3(np.linalg.inv(transform_new).dot(vec3tovec4(child.corrected_position)))
				vec = vec/np.linalg.norm(vec)
				rot = np.array(child.offset)
				rot = rot/np.linalg.norm(rot)

				finapos.append(vec)
				initpos.append(rot)


			ta = self.find_rotation(initpos,finapos)
			# print(ta)
			trans = rotation_z(ta[0]).dot(rotation_x(ta[1])).dot(rotation_y(ta[2]))
			# print(trans)
			transform_new = transform_new.dot(trans)
			node.corrected_dof_values = [ta[0],ta[1],ta[2]]
		for child in node.children:
			self.set_corrected_dofs_helper(child,dc(transform_new))

	def find_rotation(self,initpos,finapos):
		#Ax = 0
		length = len(initpos)
		A = np.zeros((length*3,10))
		for i in range(length):
			A[3*i,0] = initpos[i][0]
			A[3*i,1] = initpos[i][1]
			A[3*i,2] = initpos[i][2]

			A[3*i+1,3] = initpos[i][0]
			A[3*i+1,4] = initpos[i][1]
			A[3*i+1,5] = initpos[i][2]

			A[3*i+2,6] = initpos[i][0]
			A[3*i+2,7] = initpos[i][1]
			A[3*i+2,8] = initpos[i][2]

			A[3*i,9] = -finapos[i][0]
			A[3*i+1,9] = -finapos[i][1]
			A[3*i+2,9] = -finapos[i][2]

		w,v = eig(np.transpose(A).dot(A),k=1,which = 'SM')
		vec = np.real(v.transpose()[0])

		# print(np.transpose(A).dot(A))

		vec = vec/vec[9]
		vec = vec[:9]
		interim = np.reshape(vec,(3,3))
		s,v,d = svd(interim)
		rot = s.dot(d)
		# print(vec)
		if(det(rot)<0):
			rot = s.dot(np.diag([1,1,-1]).dot(d))

		# print(rot)
		x,y,z = self.rotm2eul(rot)
		return [z,x,y]

	def rotm2eul(self,rotm):
		x = np.arcsin(rotm[2,1])
		x = [x]
		if(x[-1]>0):
			x.append(x[-1]-np.pi)
		else:
			x.append(x[-1]+np.pi)
		# print(x)

		y = []
		for X in x:
			y.append(np.arcsin(-rotm[2,0]/np.cos(X)))
			if(y[-1]>0):
				y.append(y[-1]-np.pi)
			else:
				y.append(y[-1]+np.pi)

		z = []
		for X in x:
			val = rotm[1,1]/np.cos(X)
			if(val>1):
				val = 0.99999999999999999999
			if(val<-1):
				val = -0.99999999999999999999
			z.append(np.arccos(val))
			z.append(-np.arccos(val))

		xyz = []
		for Y in y:
			for Z in z:
				for X in x:
					mat = rotation_z(np.rad2deg(Z)).dot(rotation_x(np.rad2deg(X)).dot(rotation_y(np.rad2deg(Y))))[:3,:3]
					if(norm2(mat-rotm)<0.0001):
						xyz.append((np.rad2deg(X),np.rad2deg(Y),np.rad2deg(Z)))

		return xyz[0]


	def set_corrected_dofs_helper_fixedy(self,node,transform):
		transform_new = dc(transform)
		node.corrected_transform = transform_new
		# node.corrected_position = node.corrected_transform.dot(vec3tovec4(node.offset))
		transform_new = dc(node.corrected_transform.dot(translation(node.offset)))
		if(node.name=="End Site"):
			node.corrected_dof_values = []
			return
		if(node.depth==0):
		# if(True):
			inter_transform = np.identity(4)
			for i in range(len(node.dofs)):
				dof = node.dofs[i]
				dof_val = node.dof_values[i]
				rot = np.identity(4)
				if(dof=="Xposition"):
					rot = translation((dof_val,0,0))
				elif(dof=="Yposition"):
					rot = translation((0,dof_val,0))
				elif(dof=="Zposition"):
					rot = translation((0,0,dof_val))
				elif(dof=="Xrotation"):
					rot = rotation_x(dof_val)
				elif(dof=="Yrotation"):
					rot = rotation_y(dof_val)
				elif(dof=="Zrotation"):
					rot = rotation_z(dof_val)
				inter_transform = inter_transform.dot(rot)
				# transform_new = transform_new.dot(rot)

			transform_new = transform_new.dot(inter_transform)
			node.corrected_dof_values = node.dof_values[:]
			node.noised_position = node.position[:3]
			node.corrected_position = node.noised_position[:]
			for child in node.children:
				child.noised_position = child.position[:3]
				child.corrected_position = child.noised_position[:]
			# print(inter_transform)
			# transform_new2 = transform_new
			# print(inter_transform)
			# print("inter_transform",inter_transform)
		# if(True):
		else:
			TA = np.empty((len(node.children),3))
			i=0
			for child in node.children:
				vec = vec4tovec3(np.linalg.inv(transform_new).dot(vec3tovec4(child.corrected_position)))
				vec = vec/np.linalg.norm(vec)
				rot = np.array(child.offset)
				rot = rot/np.linalg.norm(rot)

				ta,ret_val = find_transform(rot,vec,node.dof_values[-1],node.dof_values)

				if(ret_val==-1):
					child.noised_position = child.position[:3]
					diff = np.array(child.noised_position)-np.array(node.corrected_position)
					lengthratio = norm2(diff)/norm2(child.offset)
					diff = diff/lengthratio
					diff = diff+np.array(node.corrected_position)
					child.corrected_position = diff.tolist()

				TA[i,:] = np.array([ta[0],ta[1],ta[2]])
				i+=1

			ta = np.mean(TA,axis=0)
			trans = rotation_z(ta[0]).dot(rotation_x(ta[1])).dot(rotation_y(ta[2]))
			transform_new = transform_new.dot(trans)
			node.corrected_dof_values = [ta[0],ta[1],ta[2]]
		for child in node.children:
			self.set_corrected_dofs_helper(child,dc(transform_new))



	def set_corrected_dofs_helper_imp_sampling(self,node,transform,noise_std):
		transform_new = dc(transform)
		node.corrected_transform = transform_new
		# node.corrected_position = node.corrected_transform.dot(vec3tovec4(node.offset))
		if(node.name=="End Site"):
			return

		threshold = 10
		while True:
			length = norm2(node.children[0].offset)
			transform_new = dc(node.corrected_transform.dot(translation(node.offset)))
			if(len(node.dof_values)!=6):
				node.corrected_dof_values = np.random.normal(node.dof_values,np.rad2deg(noise_std/length))
			else:
				node.corrected_dof_values = node.dof_values[:]
			
			for i in range(len(node.dofs)):
				dof = node.dofs[i]
				dof_val = node.corrected_dof_values[i]
				if(dof=="Xposition"):
					trans = translation((dof_val,0,0))
					transform_new = transform_new.dot(trans)
				elif(dof=="Yposition"):
					trans = translation((0,dof_val,0))
					transform_new = transform_new.dot(trans)
				elif(dof=="Zposition"):
					trans = translation((0,0,dof_val))
					transform_new = transform_new.dot(trans)
				elif(dof=="Xrotation"):
					rot = rotation_x(dof_val)
					transform_new = transform_new.dot(rot)
				elif(dof=="Yrotation"):
					rot = rotation_y(dof_val)
					transform_new = transform_new.dot(rot)
				elif(dof=="Zrotation"):
					rot = rotation_z(dof_val)
					transform_new = transform_new.dot(rot)

			break_signal = True

			for child in node.children:
				if(not norm2(vec4tovec3(transform_new.dot(vec3tovec4(child.offset)))-child.corrected_position)<threshold):
					break_signal = False
				else:
					print(child.name)
					print(vec4tovec3(transform_new.dot(vec3tovec4(child.offset))))
					print(child.corrected_position)


			if(break_signal):
				break
		for child in node.children:
			self.set_corrected_dofs_helper(child,dc(transform_new),noise_std)

def readHeaderWithTabs(BVHHeader):
	with open (BVHHeader, "r") as BVHHFile:
		header=BVHHFile.readlines()
	return header

def readHeader(BVHHeader):
	header = readHeaderWithTabs(BVHHeader)
	return [t.rstrip().strip() for t in header]

def read_bvh_data(fileName):
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

def vec3tovec4(vec3):
	return np.array([vec3[0],vec3[1],vec3[2],1])

def vec4tovec3(vec4):
	return np.array([vec4[0]/vec4[3],vec4[1]/vec4[3],vec4[2]/vec4[3]])


def rotation_axis(axis,angle):
	axis = np.array(axis)
	axis = axis/np.linalg.norm(axis)
	angle = np.radians(angle)
	x,y,z = axis[0],axis[1],axis[2]
	c = np.cos(angle)
	s = np.sin(angle)
	rot = np.identity(4)
	rot[0,0] = c + x*x*(1-c)
	rot[0,1] = x*y*(1-c)-z*s
	rot[0,2] = x*z*(1-c)+y*s
	rot[1,0] = y*x*(1-c)+z*s
	rot[1,1] = c+y*y*(1-c)
	rot[1,2] = y*z*(1-c)-x*s
	rot[2,0] = z*x*(1-c)-y*s
	rot[2,1] = z*y*(1-c)+x*s
	rot[2,2] = c+z*z*(1-c)

	return rot


def find_transform(p1,p2,yrot,actual):
	tuples = []
	yrot = np.radians(yrot)
	cy = np.cos(yrot)
	sy = np.sin(yrot)
	# p2[2] = -p1[0]*sy*cx+p1[1]*sx+p1[2]*cy*cx
	# cx*(p1[2]*cy-p1[0]*sy) + sx*(p1[1]) = p2[2]
	R = np.sqrt(p1[1]**2+(p1[2]*cy-p1[0]*sy)**2)
	alpha = np.arctan2((-p1[0]*sy+p1[2]*cy),p1[1])

	# print(p2[2])
	# print(p1,p2,yrot,R)
	if(R==0):
		return (actual[0],actual[1],actual[2]),-1
	if(p2[2]/R>1.0 or p2[2]/R<-1.0 ):
		return (actual[0],actual[1],actual[2]),-1
	theta_alpha = np.arcsin(p2[2]/R)

	theta_alpha = (theta_alpha,np.pi-theta_alpha)

	theta = theta_alpha-alpha
	for xrot in theta:
		if(xrot>np.pi): xrot = xrot-2*np.pi
		sx = np.sin(xrot)
		cx = np.cos(xrot)
		# print(3*cx-4*sx-2)
		# sz*(cy*sx*p1[2]-sx*sy*p1[0]-cx*p1[1])+cz*(cy*p1[0]+sy*p1[2]) = p2[0]
		R = np.sqrt((cy*p1[0]+sy*p1[2])**2+(sx*sy*p1[0]-cy*sx*p1[2]+cx*p1[1])**2)

		alpha = np.arctan2((cy*p1[0]+sy*p1[2]),(cy*sx*p1[2]-sx*sy*p1[0]-cx*p1[1]))

		theta_alpha = np.arcsin(p2[0]/R)
		if(R==0):
			return (actual[0],actual[1],actual[2]),0
		if(p2[2]/R>1.0 or p2[2]/R<-1.0 ):
			return (actual[0],actual[1],actual[2]),0
		theta_alpha = (theta_alpha,np.pi-theta_alpha)
		thetaz = theta_alpha-alpha
		# print(thetaz)
		for zrot in thetaz:
			if(zrot>np.pi): zrot = zrot-2*np.pi
			sz = np.sin(zrot)
			cz = np.cos(zrot)
			# print(sz,cz)
			if p2[1]-(p1[0]*(cy*sz+cz*sx*sy)+p1[1]*(cz*cx)+p1[2]*(sz*sy-cz*cy*sx))<0.0001 and\
			p2[1]-(p1[0]*(cy*sz+cz*sx*sy)+p1[1]*(cz*cx)+p1[2]*(sz*sy-cz*cy*sx))>-0.0001:
				tuples.append((np.rad2deg(zrot),np.rad2deg(xrot),np.rad2deg(yrot)))
				# print(vec4tovec3(rotation_z(zrot).dot(rotation_x(xrot).dot(rotation_y(yrot))).dot(vec3tovec4(p1)))-p2)
	# print(len(tuples))
	if(mod(actual[0]-tuples[0][0])<mod(actual[0]-tuples[1][0])):
		return tuples[0],0
	else:
		return tuples[1],0

def mod(num):
	if(num<0): return -num
	return num

def rotation_x(angle):
	angle = np.radians(angle)
	c = np.cos(angle)
	s = np.sin(angle)
	rot = np.identity(4)
	rot[1,1],rot[1,2] = c, -s
	rot[2,1],rot[2,2] = s, c
	return rot

def rotation_y(angle):
	angle = np.radians(angle)
	c = np.cos(angle)
	s = np.sin(angle)
	rot = np.identity(4)
	rot[2,2],rot[2,0] = c, -s
	rot[0,2],rot[0,0] = s, c
	return rot

def rotation_z(angle):
	angle = np.radians(angle)
	c = np.cos(angle)
	s = np.sin(angle)
	rot = np.identity(4)
	rot[0,0],rot[0,1] = c, -s
	rot[1,0],rot[1,1] = s, c
	return rot

def translation(xyz):
	tran = np.identity(4)
	tran[0,3] = xyz[0]
	tran[1,3] = xyz[1]
	tran[2,3] = xyz[2]
	return tran





