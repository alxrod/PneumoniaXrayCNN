import os

directory = raw_input("What directory do you want to rename? ")
nameList = os.listdir(directory)
print nameList

newName = raw_input("What is the new name you want to call these files? ")
count = 0
for l in nameList:
	print l
	os.rename(l, (newName+"_"+str(count)))
	count+=1