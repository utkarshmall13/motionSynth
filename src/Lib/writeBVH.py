def writeFile(fileName, BVHMotion, BVHHeader, frame_rate):
	with open (BVHHeader, "r") as BVHHFile:
		header=BVHHFile.readlines()
	with open (fileName+'.bvh', "w") as outputFile:
		outputFile.writelines(header)
		outputFile.write('MOTION\n')
		outputFile.write('Frames: '+str((len(BVHMotion)))+'\n')
		outputFile.write('Frame Time:	'+str(frame_rate)+'\n')
		for i in range(len(BVHMotion)):
			line=''
			for x in range(len(BVHMotion[0])):
				line+=str(round(float(BVHMotion[i][x]),4))+' '
			line+='\n'
			outputFile.write(line)
	return
