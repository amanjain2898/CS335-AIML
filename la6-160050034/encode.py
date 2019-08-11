import sys

file_name = sys.argv[1]
prob=1

if not sys.argv[-1] == file_name:
	prob = float(sys.argv[2])

f = open(file_name,'r')

maze = []
St = 0
Es = []
A = 4
curr=0
dicty = {}
gamma = 1
j1=0
prob = 1
for line in f:
	list1 = line.split()
	col = len(list1)
	list2 = [0 for i in range(col)]
	for i in range(len(list1)):
		list2[i] = int(list1[i])

		if list2[i] == 0:
			dicty[(j1,i)] = curr
			curr += 1
		if list2[i] == 2:
			dicty[(j1,i)] = curr
			St = curr
			curr += 1
		if list2[i] == 3:
			dicty[(j1,i)] = curr
			Es.append(curr)	
			curr += 1

	j1 += 1
	maze.append(list2)

print("numStates", curr)
print("numActions 4")
print("start",St)
str1 = "end "
for i in range(len(Es)):
	str1 += str(Es[i]) + " "

print(str1)	

for i in range(len(maze)-1):
	for j in range(len(maze[0])-1):
		if maze[i][j] == 0 or maze[i][j] == 2:
			if not (i-1) < 0:
				if maze[i-1][j] == 3:
					print("transitions",dicty[(i,j)],1,dicty[(i-1,j)],0,prob)
				if maze[i-1][j] == 0 or maze[i-1][j] == 2:
					print("transitions",dicty[(i,j)],1,dicty[(i-1,j)],-1,prob)

			if not (j-1) < 0:
				if maze[i][j-1] == 3:
					print("transitions",dicty[(i,j)],0,dicty[(i,j-1)],0,prob)
				if maze[i][j-1] == 0 or maze[i][j-1] == 2:
					print("transitions",dicty[(i,j)],0,dicty[(i,j-1)],-1,prob)		

			if not (j+1) >= len(maze[0]):
				if maze[i][j+1] == 3:
					print("transitions",dicty[(i,j)],2,dicty[(i,j+1)],0,prob)
				if maze[i][j+1] == 0 or maze[i][j+1] == 2:
					print("transitions",dicty[(i,j)],2,dicty[(i,j+1)],-1,prob)

			if not (i+1) >= len(maze[0]):
				if maze[i+1][j] == 3:
					print("transitions",dicty[(i,j)],3,dicty[(i+1,j)],0,prob)
				if maze[i+1][j] == 0 or maze[i+1][j] == 2:
					print("transitions",dicty[(i,j)],3,dicty[(i+1,j)],-1,prob)


print("discount",gamma)