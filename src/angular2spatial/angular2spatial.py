import argparse
import sys
import numpy as np
import os
from os.path import join,isdir

sys.path.append('../Lib/')
import bvh23d
import bvhHandler as bh

parser = argparse.ArgumentParser(description='Argument Parser Commands')
parser.add_argument('-id','--input-dir',type=str,required = True)
parser.add_argument('-od','--output-dir',type=str,required = True)
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

############################################################################
# List all files in id
subdirs = os.listdir(input_dir)
files = []
for subdir in subdirs:
	files+=[(subdir,tmp) for tmp in os.listdir(join(input_dir,subdir))]

print(files)
############################################################################

BVHHeader = "../Lib/BVHHeader96.txt"
header = bvh23d.readHeader(BVHHeader)
tree = bvh23d.Tree(header)

for k in range(len(files)):
	frames = bh.readBVH(join(input_dir, files[k][0], files[k][1]))
	frames_new = []
	pts = []
	for i in range(0,len(frames)):
		tree.set_dof_values(frames[i])
		tree.set_transformations_values()
		frame = []
		pos = tree.get_position()
		for tup in pos:
			frame+=list(tup)
			pts.append(list(tup))
		frames_new.append(frame[3:])
	if(not isdir(join(output_dir, files[k][0]))):
		os.mkdir(join(output_dir, files[k][0]))

	np.savetxt(join(output_dir, files[k][0], files[k][1]),frames_new,fmt='%.4f',delimiter=" ")
	print(k,"/",len(files))
