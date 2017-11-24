import math
import numpy as np
from functools import reduce
import sys
from io import StringIO

_FLOAT_EPS_4 = np.finfo(float).eps * 4.0

def euler2mat(z=0, y=0, x=0):
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)

def mat2euler(M, cy_thresh=None):
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

def ZYX2YXZ(z,y,x):
	mat=euler2mat(-math.radians(z),math.radians(y),math.radians(x))	
	tmp=mat[0][0]
	mat[0][0]=mat[2][2]
	mat[2][2]=mat[1][1]
	mat[1][1]=tmp
	
	tmp=mat[0][1]
	mat[0][1]=mat[2][0]
	mat[2][0]=mat[1][2]
	mat[1][2]=tmp
	
	tmp=mat[0][2]
	mat[0][2]=mat[2][1]
	mat[2][1]=mat[1][0]
	mat[1][0]=tmp

	y,z,x=mat2euler(mat)
	return math.degrees(y),math.degrees(z),-math.degrees(x)

def YXZ2ZYX(z,y,x):
	mat=euler2mat(math.radians(z),-math.radians(y),math.radians(x))	
	tmp=mat[0][0]
	mat[0][0]=mat[1][1]
	mat[1][1]=mat[2][2]
	mat[2][2]=tmp
	
	tmp=mat[0][1]
	mat[0][1]=mat[1][2]
	mat[1][2]=mat[2][0]
	mat[2][0]=tmp
	
	tmp=mat[0][2]
	mat[0][2]=mat[1][0]
	mat[1][0]=mat[2][1]
	mat[2][1]=tmp

	y,z,x=mat2euler(mat)
	return math.degrees(y),math.degrees(z),-math.degrees(x)
