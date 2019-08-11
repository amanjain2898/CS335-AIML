import sys
import copy

file_name = sys.argv[1]
f = open(file_name,'r')

S = int(next(f).split()[1])
A = int(next(f).split()[1])

ST = int(next(f).split()[1])

T_m = [-1 for i in range(S)]

end_list = next(f).split()
for i in range(1,len(end_list)):
	if int(end_list[i]) == -1:
		break
	else:
		T_m[int(end_list[i])] = 1


gamma = 0

Tr = [[{} for i in range(A)] for j in range(S)]
Rd = [[{} for i in range(A)] for j in range(S)]

while 1:
	list1 = next(f).split()
	length1 = len(list1)
	if length1 == 2:
		gamma = float(list1[1])
		break
	else:
		Tr[int(list1[1])][int(list1[2])][int(list1[3])] = float(list1[5])
		Rd[int(list1[1])][int(list1[2])][int(list1[3])] = float(list1[4])


##################################################################################

V = [0 for i in range(S)]
PI = [-1 for i in range(S)]
iterations = 0

while 1:
	iterations += 1
	prev = copy.deepcopy(V)
	for i in range(S):
		max1 = -float("inf")
		if not T_m[i] == 1:
			for j in range(A):
				sum1 = 0
				if not Tr[i][j]:
					continue
				for k,x in Tr[i][j].items():
						sum1 += Tr[i][j][k]*(Rd[i][j][k] + gamma*prev[k])
				if sum1 > max1:
					V[i] = sum1
					PI[i] = j
					max1 = sum1
	flag = 1
	for i in range(S):
		if abs(V[i] - prev[i]) > 1e-16:
			flag = 0

	if flag == 1:
		break


for i in range(S):
	print(V[i],PI[i])

print("iterations",iterations)
