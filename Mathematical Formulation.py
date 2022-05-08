# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 09:51:56 2022

@author: mikel
"""
import numpy as np

#sed = np.loadtxt('C:\\Users\\mikel\\Desktop\\Thesis\\Large instance\\31-128-3-33.dat', unpack = True)

import gurobipy as gp

Statement = input ('Are soft time windows being considered? (Yes/No): ')
if Statement =='YES' or Statement=='Yes' or Statement=='yes': 
    soft = True
else:
    soft = False

# from n to l needs to be changed (as well as t)
n = 7
K = 3
P_e = 10
P_l = 15
C = 50
T_max = 20
cluster_list = [[0,n+1],[1,2],[3,4],[5,6,7]]
h = [0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,0]
e = [0,1,3,2,0.5,5,3,0,0]
l = [T_max,4,9,18,9.5,15,17,3,T_max]

curlyK = [i for i in range(K+1)]
V = [i for i in range(n+2)]
Vm0 = [i for i in range(1,n+1)]

# Matrix of driving times t_ij
t = [[ 0 ,	1.39,	1.36,	6.84,	5.33,	6.98,	5.04,	1.97, 0],
     [1.39,	 0, 	2.25,	4.15,	6.19,	11.22,	5.55,	3, 1.39],
[1.36,	2.25,	 0 ,	1.05,	5.09,	8.05,	5.38,	3.17, 1.36],
[6.84,	4.15,	1.05,	 0, 	4.53,	11.79,	9.63,	8.18, 6.84],
[5.33,	6.19,	5.09,	4.53,	 0, 	5,	4.33,	3.13, 5.33],
[6.98,	11.22,	8.05,	11.79,	5,	 0, 	4.48,	5.01, 6.98],
[5.04,	5.55,	5.38,	9.63,	4.33,	4.48,	 0, 	1.5, 5.04],
[1.97,	3,	3.17,	8.18,	3.13,	5.01,	1.5,	 0, 1.97],
[0, 1.39,1.36,6.84,5.33,6.96,5.04,1.97,0]]                
   
#Matrix of driving costs c_ij
c = [[[0] for i in range(n+2)] for j in range(n+2)]
for i in range(n+2):
    for j in range(n+2):
        c[i][j] = C*t[i][j]
 
#Printing the model parameters
print('Soft Time Windows:', soft)
print('Total Number of Customer Nodes:', n)
print('Early Penalty:', P_e)
print('Late Penalty:', P_l)
print('Due Time of Depot:', l[0])

#Defining the model
GTSPTW = gp.Model(name='Generalized Travelling Salesman Problem with Time Windows')

#Defining the decision variables x_ij
x = {}
for i in V:
    for j in V:
        if (i != n+1) and (j != 0) and (j != i):       
              x[i,j]= GTSPTW.addVar(vtype=gp.GRB.BINARY, name='x' + str(i) + ',' + str(j))

#Defining the decision variables Z_i 
Z = {}
for i in V:
    Z[i] = GTSPTW.addVar(vtype=gp.GRB.CONTINUOUS, lb =0, name='Z' + str(i))

# Defining the decision variables for earliness/lateness (soft time windows)  
if (soft == True):
    early = {}
    late = {} 
    for i in Vm0:
        early[i] = GTSPTW.addVar(vtype=gp.GRB.CONTINUOUS, lb =0, name='early' + str(i))
        late[i] = GTSPTW.addVar(vtype=gp.GRB.CONTINUOUS, lb =0, name='late' + str(i))
    
#Constraint Set 1 (Excluding Arcs linking Nodes within Same Cluster)
for i in V:
    if (i != n+1):
        for k in curlyK:
                if i in cluster_list[k]:
                    cluster_index_i = k
                    break 
        for j in V:
            if (j != 0) and (j != i) and (j in cluster_list[cluster_index_i]): 
                  GTSPTW.addConstr(x[i,j] == 0)
                  
# Constraint Set 2 (Route Continuity)
for i in Vm0:
    GTSPTW.addConstr(gp.quicksum(x[j,i] for j in V if (j != i) and (j != n+1)) - gp.quicksum(x[i,j] for j in V if (j != 0) and (j != i)) == 0)
                                     
#Constraint Set 3 (Customer Cluster Visits)
for k in curlyK:
   if (k != 0):
       GTSPTW.addConstr(gp.quicksum(x[i,j] for i in cluster_list[k] for j in V if (j != 0) and (j != i)) == 1)

#Constraint Set 4 (Depot Visits)
GTSPTW.addConstr(gp.quicksum(x[0,j] for j in Vm0) == 1)
GTSPTW.addConstr(gp.quicksum(x[i,n+1] for i in Vm0)== 1)
    
#Constraint Set 5 (Hard Time Windows)  
if (soft == False):
     for i in Vm0:
         Z[i] >= e[i]
         Z[i] <= l[i]
         #GTSPTW.addConstr(e[i]*gp.quicksum(x[i,j] for j in V if (j != 0) and (j != i)) <= Z[i])
         #GTSPTW.addConstr(l[i]*gp.quicksum(x[i,j] for j in V if (j != 0) and (j != i)) >= Z[i])
      
#Constraint Set 6 (Route Time Limit)
GTSPTW.addConstr(e[0] <= Z[0])
GTSPTW.addConstr(Z[0] <= l[0])
GTSPTW.addConstr(e[n+1] <= Z[n+1])
GTSPTW.addConstr(Z[n+1] <= l[n+1])

#Constraint Set 7 (Subtour Elimination)
for i in V:
    for j in V:
        if (i != n+1) and (j != 0) and (j != i):   
            if (soft==True):
                GTSPTW.addConstr(Z[i]+t[i][j]+h[i]+(T_max +t[i][j]+h[i])*(x[i,j]-1)<= Z[j])
                GTSPTW.addConstr(Z[j]<=Z[i]+t[i][j]+h[i]+(t[i][j]+h[i]-T_max)*(x[i,j]-1))
            else:
                GTSPTW.addConstr(Z[i]+t[i][j]+h[i]+(l[i] +t[i][j]+h[i]-e[j])*(x[i,j]-1)<= Z[j])

#Constraints required for earlines/lateness (soft time windows)
if (soft==True):
    for i in Vm0: 
        GTSPTW.addConstr(early[i] >= e[i] - Z[i])
        GTSPTW.addConstr(late[i] >= Z[i] - l[i])
    
#Objective Function
objective = gp.quicksum(c[i][j]*x[i,j] for i in V for j in V if (i != n+1) and (j != 0) and (j != i))
if (soft==True):
    for i in Vm0:     
        objective += P_e*early[i] + P_l*late[i]

#Setting Minimization
GTSPTW.ModelSense = gp.GRB.MINIMIZE
GTSPTW.setObjective(objective)
#MIPGap is set to 0 to return the global optimum
GTSPTW.setParam('MIPGap',0)      # default value is 1e-4
GTSPTW.setParam('TimeLimit',4*60*60) #in seconds 4hrs
#Optimizing the model
GTSPTW.optimize()

#Displaying the optimal solution
for v in GTSPTW.getVars():
    if (v.x != 0): #v.x is the optimal value of the variable v
        print(v.varName, v.x)  #printing variable name followed by optimal value

for i in Vm0: 
     if Z[i].x < e[i]:
         print(i,e[i]-Z[i].x,"early") #nothing will be displayed
     elif Z[i].x > l[i]:
         print(i,Z[i].x-l[i],"late")
