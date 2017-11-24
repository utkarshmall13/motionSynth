import bvh23d
import bvhHandler as bh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-fn","--filename",required = True)
args = parser.parse_args()
filename = args.filename


header = bvh23d.readHeader("BVHHeader.txt")
tree = bvh23d.Tree(header)

frames = bh.readBVH(filename)

frames_new = []
for i in range(0,len(frames)):
	tree.set_dof_values(frames[i])
	tree.set_transformations_values()
	ret = tree.get_position()
	#output format for point cloud viewer
	# for reti in ret:
	# 	print(str(reti[0])+" "+str(reti[1])+" "+str(reti[2])+" "+"1 1 1")
	frames_new.append(ret)	

print(frames_new)


