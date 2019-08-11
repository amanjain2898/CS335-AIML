import sys
import numpy as np
import random

random.seed(0)

file_name = sys.argv[1]
p_fname = sys.argv[2]
prob=1
if not sys.argv[-1] == p_fname:
	prob = float(sys.argv[3])

f = open(file_name,'r')
g = open(p_fname,'r')

maze = []
St = 0
Es = []
A = 4
curr=0
dicty = {}
coord = {}

j=0
for line in f:
	list1 = line.split()
	col = len(list1)
	list2 = [0 for i in range(col)]
	for i in range(len(list1)):
		list2[i] = int(list1[i])

		if list2[i] == 0:
			dicty[(j,i)] = curr
			coord[curr] = (j,i)
			curr += 1
		if list2[i] == 2:
			dicty[(j,i)] = curr
			coord[curr] = (j,i)
			St = curr
			curr += 1
		if list2[i] == 3:
			dicty[(j,i)] = curr
			coord[curr] = (j,i)
			Es.append(curr)
			curr += 1

	j += 1
	maze.append(list2)

gamma = 1

optimal_actions = {}
cnt=0
for line in g:
	list1 = line.split()
	if list1[0] == "iterations":
		break
	if list1[1] == "-1":
		optimal_actions[cnt] = "end"
	else:
		optimal_actions[cnt] = int(list1[1])
	cnt += 1

state = St
path = ""

while(1):
	if not optimal_actions[state] == "end":
		if optimal_actions[state] == 0:
			x1 = int(coord[state][0])
			y1 = int(coord[state][1])
			list3 = []
			if not dicty.get((x1+1,y1)) == None:
				list3.append(3)
			if not dicty.get((x1-1,y1)) == None:
				list3.append(1)
			if not dicty.get((x1,y1-1)) == None:
				list3.append(0)
			if not dicty.get((x1,y1+1)) == None:
				list3.append(2)

			weights = []
			for k1 in range(len(list3)):
				weights.append((1-prob)*1.0/len(list3))	

			list3.append(0)
			weights.append(prob)
			# print(list3)
			# print(weights)
			rand = np.random.choice(list3,p=weights)

			if rand == 0:
				path += "W "
			elif rand == 1:
				path += "N "
			elif rand == 2:
				path += "E "
			elif rand == 3:
				path += "S "

			x_coor = int(coord[state][0])
			y_coor = int(coord[state][1])

			if rand == 1:
				state = dicty[(x_coor-1,y_coor)]
			elif rand == 0:
				state = dicty[(x_coor,y_coor-1)]
			elif rand == 3:
				state = dicty[(x_coor+1,y_coor)]
			elif rand == 2:
				state = dicty[(x_coor,y_coor+1)]

		elif optimal_actions[state] == 1:
			x1 = int(coord[state][0])
			y1 = int(coord[state][1])
			list3 = []
			if not dicty.get((x1+1,y1)) == None:
				list3.append(3)
			if not dicty.get((x1-1,y1)) == None:
				list3.append(1)
			if not dicty.get((x1,y1-1)) == None:
				list3.append(0)
			if not dicty.get((x1,y1+1)) == None:
				list3.append(2)

			weights = []
			for k1 in range(len(list3)):
				weights.append((1-prob)*1.0/len(list3))	

			list3.append(1)
			weights.append(prob)
			# print(list3)
			# print(weights)
			rand = np.random.choice(list3,p=weights)

			if rand == 0:
				path += "W "
			elif rand == 1:
				path += "N "
			elif rand == 2:
				path += "E "
			elif rand == 3:
				path += "S "

			x_coor = int(coord[state][0])
			y_coor = int(coord[state][1])

			if rand == 1:
				state = dicty[(x_coor-1,y_coor)]
			elif rand == 0:
				state = dicty[(x_coor,y_coor-1)]
			elif rand == 3:
				state = dicty[(x_coor+1,y_coor)]
			elif rand == 2:
				state = dicty[(x_coor,y_coor+1)]

		elif optimal_actions[state] == 2:
			x1 = int(coord[state][0])
			y1 = int(coord[state][1])
			list3 = []
			if not dicty.get((x1+1,y1)) == None:
				list3.append(3)
			if not dicty.get((x1-1,y1)) == None:
				list3.append(1)
			if not dicty.get((x1,y1-1)) == None:
				list3.append(0)
			if not dicty.get((x1,y1+1)) == None:
				list3.append(2)

			weights = []
			for k1 in range(len(list3)):
				weights.append((1-prob)*1.0/len(list3))	

			list3.append(2)
			weights.append(prob)
			# print(list3)
			# print(weights)
			rand = np.random.choice(list3,p=weights)

			if rand == 0:
				path += "W "
			elif rand == 1:
				path += "N "
			elif rand == 2:
				path += "E "
			elif rand == 3:
				path += "S "

			x_coor = int(coord[state][0])
			y_coor = int(coord[state][1])

			if rand == 1:
				state = dicty[(x_coor-1,y_coor)]
			elif rand == 0:
				state = dicty[(x_coor,y_coor-1)]
			elif rand == 3:
				state = dicty[(x_coor+1,y_coor)]
			elif rand == 2:
				state = dicty[(x_coor,y_coor+1)]

		elif optimal_actions[state] == 3:
			x1 = int(coord[state][0])
			y1 = int(coord[state][1])
			list3 = []
			if not dicty.get((x1+1,y1)) == None:
				list3.append(3)
			if not dicty.get((x1-1,y1)) == None:
				list3.append(1)
			if not dicty.get((x1,y1-1)) == None:
				list3.append(0)
			if not dicty.get((x1,y1+1)) == None:
				list3.append(2)

			weights = []
			for k1 in range(len(list3)):
				weights.append((1-prob)*1.0/len(list3))

			list3.append(3)
			weights.append(prob)
			# print(list3)
			# print(weights)
			rand = np.random.choice(list3,p=weights)

			if rand == 0:
				path += "W "
			elif rand == 1:
				path += "N "
			elif rand == 2:
				path += "E "
			elif rand == 3:
				path += "S "

			x_coor = int(coord[state][0])
			y_coor = int(coord[state][1])

			if rand == 1:
				state = dicty[(x_coor-1,y_coor)]
			elif rand == 0:
				state = dicty[(x_coor,y_coor-1)]
			elif rand == 3:
				state = dicty[(x_coor+1,y_coor)]
			elif rand == 2:
				state = dicty[(x_coor,y_coor+1)]
 
	else:
		break

print(path)