# -*- coding: utf-8 -*-
"""
Created on Sun May  8 23:54:34 2022

@author: mikel
"""

import time
import random
import numpy as np
from ast import literal_eval
import operator
#path_to_file = ...

counter = 1
n = 0   
cluster_list = []
t = []
e = []
l = []

for line in open(path_to_file):
    data = line.rstrip("\n").split()
    if (counter == 1):  # first line
        n = int(data[1])
        cluster_list.append([0, n+1])
    elif (counter == 2):
        K = int(data[1])
    elif (counter in range(3, 3 + K)):
        cluster_list.append([int(x) for x in data[1:]])
    elif (counter in range(6+K+n, 7+K+2*n)):  
        t.append([int(x) for x in data])
    elif (counter in range(8+K+2*n, 9+K+3*n)):  
        e.append(int(data[0]))
        l.append(int(data[1]))
    counter += 1
e.append(e[0])
l.append(l[0])
T_max = l[0]

t[0].append(0)  # added a zero to first row of t
t.append(t[0])
for row in t[1:n+1]:
    # appending to each row of t the first element of that particular row
    row.append(row[0])

h_fixed = 2  # h_fixed is a fixed service time
h = [0]
for i in range(1, n+1):
    h.append(h_fixed)
h.append(0)
P_e = 10  # fixed early penalty
P_l = 15  # fixed late penalty
C = 50
Vm0 = [i for i in range(1, n+1)]
c = [[[0] for i in range(n+2)] for j in range(n+2)]
for i in range(n+2):
    for j in range(n+2):
        c[i][j] = C*t[i][j]
        


# GA parameters 
population_size =100 
elitism_rate = 0.2 #the best % of current population that will be kept in next population 
number_of_elites = int(elitism_rate*population_size)
mutation_rate = 0.4
  # % of non-elites that will be mutated
number_of_mutations = int(mutation_rate*(population_size - number_of_elites)) # int - to round value down
number_of_generations=500




#  checking feasibility of chromosome
def check_feasibility(chromosome):
    route_time = 0
    for i in range(K+1):
        route_time = route_time + t[chromosome[i]][chromosome[i+1]]
    route_time = route_time + K*h_fixed
    if (e[0] + route_time <= T_max):
        return True
    else:
        return False


# creating one chromosome
def New_Chromosome():
    flag = False # flag keeps record of whether a solution is feasible or not    
    while flag == False:
        # identify which customer nodes shall be visited 
        chosen_nodes = []
        for i in range(1,K+1): # depot cluster not considered
            chosen_nodes.append(random.choice(cluster_list[i]))
        #initializing
        current_node = 0
        next_node = 0
        min_distance= 1000000
        chromosome =[0]
        route_time = 0
        
        # procedure to find the closest neighbour to the current node, add it to chromosome and remove it from chosen_nodes
        while len(chosen_nodes) > 1: # more than one chosen node still to be added to chromosome
            for i in chosen_nodes:
                if t[current_node][i] < min_distance:
                    min_distance = t[current_node][i]
                    next_node = i
            # adding the time from the current node to the next node 
            route_time = route_time + t[current_node][next_node]
            current_node = next_node
            min_distance = 1000000
            chosen_nodes.remove(next_node)
            chromosome.append(next_node)
        chromosome.append(chosen_nodes[0]) # when one chosen node is left, we must visit this last chosen node
        route_time = route_time + t[current_node][chosen_nodes[0]]
        chromosome.append(n+1) # visiting the end depot
        route_time = route_time + t[chosen_nodes[0]][n+1]
        route_time = route_time + K*h_fixed   
        if (e[0] + route_time <= T_max):      
            flag = True    
    return chromosome


# creating new sorted population of chromosomes
def New_Population():
    population = {}  # creating a dictionary
    for i in range(population_size):
        chromosome = New_Chromosome()
        while (str(chromosome) in population):  # to ensure no duplicates in population
            chromosome = New_Chromosome()
        population[str(chromosome)] = calc_fitness(chromosome)
        all_chromosomes[str(chromosome)] = population[str(chromosome)]      
    population = dict(sorted(population.items(), key=operator.itemgetter(1), reverse=True))  # sorting initial population
    return population


# calculating fitness value of chromosome
# takes a chromosome and returns its best total cost and best start time
def calc_fitness(chromosome):
    fitness = 0
    for i in range(K+1):
        fitness = fitness + c[chromosome[i]][chromosome[i+1]]
    route_time = int(fitness/C) + K*h_fixed  # route time of chromosome
    lowest_violation = 1000000
    s_best = e[0]            # initializing best start time for the route
    base_arrival_times = []  # list of all arrival times (if route starts at 0)
    arrival_time = 0
    for i in range(K):
        arrival_time = arrival_time + h[chromosome[i]]+t[chromosome[i]][chromosome[i+1]]
        base_arrival_times.append(arrival_time)
    for s in range(e[0], (T_max-route_time)+1):     # s = a possible start time for the route
        new_arrival_times = [x+s for x in base_arrival_times]
        violation = 0       # total violation cost for start time s
        for i in range(K):
            if new_arrival_times[i] < e[chromosome[i+1]]:    # arrives early
                violation = violation+P_e*(e[chromosome[i+1]]-new_arrival_times[i])
            elif new_arrival_times[i] > l[chromosome[i+1]]:  # arrives late
                violation = violation+P_l*(new_arrival_times[i]-l[chromosome[i+1]])
        if violation < lowest_violation:
            lowest_violation = violation
            s_best = s  # updating the start time 
    fitness = fitness+lowest_violation
    return 1/fitness, fitness, s_best    # 1/fitness is for roulette wheel selection 


# Generating the probabilities according to 1/fitness for Roulette Wheel Selection
def cumulated(population):
    pk = []     # vector of probabilities 
    qk = []     # vector of cumulative probabilities
    fitness_reciprocals = []      
    for chromosome in population:  
        fitness_reciprocals.append(population[str(chromosome)][0])   
    total_fitness = sum(fitness_reciprocals)
    for i in range(population_size):
        pk.append(fitness_reciprocals[i]/total_fitness)
    for i in range(population_size):
        cumulative = 0
        for j in range(0,i+1):
            cumulative = cumulative + pk[j]
        qk.append(cumulative)
    qk[-1] = 1   #the last element needs to be 1 exactly 
    return qk


# Selecting the parents via Roulette Wheel Selection
def selection(population):
    qk = cumulated(population)
    indices = [0,0]
    while indices[0] == indices[1]:
        # choosing two numbers at random and putting them in list selection_rand
        selection_rand = np.random.rand(2).tolist()
        indices.clear()
        for number in selection_rand:
            if number <= qk[0]:
                index = 0
            else:
                for j in range(0,population_size-1):
                    if number > qk[j] and number <= qk[j+1]:
                        index = j+1
                        break
            indices.append(index)
    # returns the two parents for crossover
    return literal_eval(list(population)[indices[0]]), literal_eval(list(population)[indices[1]])  
        

# PMX Crossover (TO SEE COMMENTS IN CAPS)
def pmx_crossover(parent_1,parent_2):
    feasible = False 
    t_start = time.time()  
    while (feasible == False and time.time()-t_start <= 0.5): 
        cutpoint = [1,K]
        while cutpoint == [1,K]:
            cutpoint = random.sample(range(1,K+1), 2)
            cutpoint.sort()
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()
        # swapping the middle parts in the children according to cutpoint
        child_1[cutpoint[0]:cutpoint[1]+1]= parent_2[cutpoint[0]:cutpoint[1]+1]
        child_2[cutpoint[0]:cutpoint[1]+1]= parent_1[cutpoint[0]:cutpoint[1]+1]
        mapping_list = []
        for i in range(cutpoint[0],cutpoint[1]+1):
            mapping_list.append([parent_1[i],parent_2[i]])
        cluster = []
        for j in range(1,K+1):
            # it will only be considered for elements outside the middle part
            if j not in range(cutpoint[0],cutpoint[1]+1):
                # procedure for child_1
                for c in cluster_list:
                    if child_1[j] in c:
                        cluster = c
                        break
                counter_1 = 0 
                for node in cluster: # the cluster where child_1[j] lies
                    counter_1 = counter_1 + child_1.count(node)
                while counter_1 > 1:    # while the cluster called 'cluster' features more than once in child_1
                    for mp in mapping_list:
                        if mp[1] in cluster: 
                            child_1[j] = mp[0]
                            break
                    for c in cluster_list:
                        if child_1[j] in c:
                            cluster = c
                            break
                    counter_1 = 0 
                    for node in cluster: 
                        counter_1 = counter_1 + child_1.count(node)               
                # procedure for child_2
                for c in cluster_list:
                    if child_2[j] in c:
                        cluster = c
                        break
                counter_2 = 0 
                for node in cluster: # the cluster where child_2[j] lies
                    counter_2 = counter_2 + child_2.count(node)
                while counter_2 > 1:   # while the cluster called 'cluster' features more than once in child_2
                    for mp in mapping_list:
                        if mp[0] in cluster: 
                            child_2[j]= mp[1]
                            break
                    for c in cluster_list:
                        if child_2[j] in c:
                            cluster = c
                            break
                    counter_2 = 0 
                    for node in cluster: 
                        counter_2 = counter_2 + child_2.count(node)
        feasible = check_feasibility(child_1) and check_feasibility(child_2)  
    return child_1, child_2, feasible  


# Required in OX Crossover
def iterFlatten(root):
    if isinstance(root, (list, tuple)):
        for element in root:
            for e in iterFlatten(element):
                yield e
    else:
        yield root 
    
    
# OX Crossover
def ox_crossover(parent_1,parent_2):
    feasible = False 
    t_start = time.time()  
    while (feasible == False and time.time()-t_start <= 0.5): 
        cutpoint = [1,K]
        while cutpoint == [1,K]:
            cutpoint = random.sample(range(1,K+1), 2)
            cutpoint.sort()
        child_1 = [0] * (K+2)
        child_2 = [0] * (K+2)
        # setting end depot
        child_1[K+1]= n+1
        child_2[K+1]= n+1
        # copying the middle part of each child from the corresponding parent
        child_1[cutpoint[0]:cutpoint[1]+1]= parent_1[cutpoint[0]:cutpoint[1]+1]
        child_2[cutpoint[0]:cutpoint[1]+1]= parent_2[cutpoint[0]:cutpoint[1]+1]
        cluster = []
        b = [[parent_1[cutpoint[1]+1:K+1],parent_1[1:cutpoint[1]+1]]]
        list_parent1= list(iterFlatten(b))  # joining lists in b together
        c = [[parent_2[cutpoint[1]+1:K+1],parent_2[1:cutpoint[1]+1]]]
        list_parent2= list(iterFlatten(c))  # joining lists in c together
        
        # if cluster where node lies already features in a child, then that node will be removed from list 
        for node in child_1[cutpoint[0]:cutpoint[1]+1]:
            # checking in which cluster the node lies
            for c in cluster_list:
                    if node in c:
                        cluster = c
                        break
            list_parent2 = [x for x in list_parent2 if x not in cluster]
        
        for node in child_2[cutpoint[0]:cutpoint[1]+1]:
            # checking in which cluster the node lies
            for c in cluster_list:
                    if node in c:
                        cluster = c
                        break
            list_parent1 = [x for x in list_parent1 if x not in cluster]
            
        #child 1
        for i in range(cutpoint[1]+1,K+1):
            child_1[i]=list_parent2[0]
            list_parent2.pop(0)
        for i in range(1,cutpoint[0]):
            child_1[i]=list_parent2[0]
            list_parent2.pop(0)
        #child 2
        for i in range(cutpoint[1]+1,K+1):
            child_2[i]=list_parent1[0]
            list_parent1.pop(0)
        for i in range(1,cutpoint[0]):
            child_2[i]=list_parent1[0]
            list_parent1.pop(0)
        feasible = check_feasibility(child_1) and check_feasibility(child_2) 
    return child_1, child_2, feasible
        

# CX Crossover
def cx_crossover(parent_1,parent_2):
    child_1 = [0] * (K+2)
    child_2 = [0] * (K+2) 
    # setting end depot
    child_1[K+1]= (n+1)
    child_2[K+1]= (n+1)
    # working on child 1
    child_1[1]=parent_1[1]
    current_index = 1
    cluster=[]
    while (current_index!= 1 or len(cluster)==0):
        node = parent_2[current_index]   # seeing what is in current index in parent 2
        # identifying the cluster in which node lies
        for c in cluster_list:
                if node in c:
                    cluster = c
                    break
        for x in cluster:
            if x in parent_1:
                current_index = parent_1.index(x)
                break
        child_1[current_index] = parent_1[current_index]
    for i in range(1,K+1):
        if child_1[i] == 0:
            child_1[i]=parent_2[i]
    # working on child 2
    child_2[1]=parent_2[1]
    current_index = 1
    cluster=[]
    while (current_index!= 1 or len(cluster)==0):
        node = parent_1[current_index]   # seeing what is in current index in parent 1
        # identifying the cluster in which node lies
        for c in cluster_list:
                if node in c:
                    cluster = c
                    break
        for x in cluster:
            if x in parent_2:
                current_index = parent_2.index(x)
                break
        child_2[current_index] = parent_2[current_index]
    for i in range(1,K+1):
        if child_2[i] == 0:
            child_2[i]=parent_1[i]             
    feasible = check_feasibility(child_1) and check_feasibility(child_2) 
    return child_1, child_2, feasible


# Best Two Selection
def bestwo(parent_1,parent_2,child_1,child_2):
    parent1fit = all_chromosomes[str(parent_1)][1]
    parent2fit = all_chromosomes[str(parent_2)][1]
    child1fit = all_chromosomes[str(child_1)][1]
    child2fit = all_chromosomes[str(child_2)][1]
    allfits = [parent1fit,parent2fit,child1fit,child2fit] # list of 4 fitness values
    bestofall = min(allfits)
    allfits.remove(bestofall)
    secondbest = min(allfits)
    if bestofall == parent1fit:
        if secondbest == parent2fit:
            return parent_1, parent_2
        elif secondbest == child1fit:
            return parent_1, child_1
        else:
            return parent_1, child_2
    elif bestofall == parent2fit:
        if secondbest == parent1fit:
            return parent_2, parent_1
        elif secondbest == child1fit:
            return parent_2, child_1
        else:
            return parent_2, child_2
    elif bestofall == child1fit:
        if secondbest == parent1fit:
            return  child_1, parent_1
        elif secondbest == parent2fit:
            return child_1, parent_2
        else:
            return child_1, child_2
    else:
        if secondbest == parent1fit:
            return child_2, parent_1 
        elif secondbest == parent2fit:
            return child_2, parent_2
        else:
            return child_2, child_1 


# Displacement Mutation
def displacement_mutation(chromosome):
    feasible = False   
    while (feasible == False):  
        cutpoint=[1,K]
        while cutpoint == [1,K]:
            cutpoint = random.choices(range(1,K+1), k=2) # selecting 2 with replacement
            cutpoint.sort() 
        list_indices=[i for i in range(1,K+2) if i not in range(cutpoint[0],cutpoint[1]+2)]
        random_index = random.choice(list_indices)
           
        portion = chromosome[cutpoint[0]:cutpoint[1]+1]
        if random_index > cutpoint[1]:
            random_index = random_index-len(portion)
        mutated_chromosome = [x for x in chromosome if x not in portion]
        mutated_chromosome[random_index:random_index]=portion
        feasible = check_feasibility(mutated_chromosome) 
    return mutated_chromosome 


# Inversion Mutation 
def inversion_mutation(chromosome):
    feasible = False  
    while (feasible == False):  
         cutpoint = random.sample(range(1,K+1),2)
         cutpoint.sort()
         mutated_chromosome= chromosome.copy()
         portion = chromosome[cutpoint[0]:cutpoint[1]+1]
         portion.reverse()
         mutated_chromosome[cutpoint[0]:cutpoint[1]+1] = portion 
         feasible = check_feasibility(mutated_chromosome) 
    return mutated_chromosome
 
    
# Inverted Displacement Mutation 
def inverted_displacement_mutation(chromosome):
    feasible = False  
    while (feasible == False):  
        cutpoint=[1,K]
        while cutpoint == [1,K]:
            cutpoint = random.choices(range(1,K+1), k=2) # selecting 2 with replacement
            cutpoint.sort() 
        list_indices=[i for i in range(1,K+2) if i not in range(cutpoint[0],cutpoint[1]+2)]
        random_index = random.choice(list_indices)
 
        portion = chromosome[cutpoint[0]:cutpoint[1]+1]
        portion.reverse()
        if random_index > cutpoint[1]:
            random_index = random_index-len(portion)
        mutated_chromosome = [x for x in chromosome if x not in portion]
        mutated_chromosome[random_index:random_index]=portion
        feasible = check_feasibility(mutated_chromosome)
    return mutated_chromosome 



# Function that gives new_population dictionary based on old_population dictionary
# old population must be sorted in non-decreasing order of fitness values (first chromosome is the best)
def nextpop(old_population): 
    new_population={}
    
    # adding elites to new_population
    for i in range(number_of_elites):
        chromosome = literal_eval(list(old_population)[i])
        new_population[str(chromosome)] = old_population[str(chromosome)]
    
    # performing crossover until population size becomes equal to population_size 
    while len(new_population) < population_size:
        parent_1,parent_2 = selection(old_population)
        #child_1,child_2,flag = pmx_crossover(parent_1,parent_2)
        child_1,child_2,flag = ox_crossover(parent_1,parent_2)
        # flag = True if both child_1 and child_2 are feasible chromosomes
        # TO CHANGE according to crossover operator (if random number == 1 take PMX)
        if (flag == True): 
            if str(child_1) not in all_chromosomes:
                all_chromosomes[str(child_1)] = calc_fitness(child_1)
            if str(child_2) not in all_chromosomes:
                all_chromosomes[str(child_2)] = calc_fitness(child_2)
            chromosome_1,chromosome_2 = bestwo(parent_1, parent_2, child_1, child_2) 
            new_population[str(chromosome_1)] = all_chromosomes[str(chromosome_1)]
            if len(new_population) < population_size: 
                new_population[str(chromosome_2)] = all_chromosomes[str(chromosome_2)]

    # performing mutation on a few selected chromosomes (that are not the elites)
    random_indices = random.sample(range(number_of_elites, population_size), number_of_mutations) # choosing indices of chromosomes to mutate
    for i in range(number_of_mutations):
        already_in_pop = True 
        while (already_in_pop == True): 
            random_index = random_indices[i]
            chromosome = literal_eval(list(new_population)[random_index])
            #mutated_chromosome = inversion_mutation(chromosome)
            mutated_chromosome = inverted_displacement_mutation(chromosome)
            # TO CHANGE according to mutation operator
            if str(mutated_chromosome) not in all_chromosomes:
                all_chromosomes[str(mutated_chromosome)] = calc_fitness(mutated_chromosome)
            new_population[str(mutated_chromosome)] = all_chromosomes[str(mutated_chromosome)]
            if (len(new_population) == population_size + (i+1)): # this implies that mutated chromosome was added to new_population
                already_in_pop = False 
    
    # performing the original chromomomes that underwent mutation
    for i in range(number_of_mutations):
        random_index = random_indices[i]
        chromosome = literal_eval(list(new_population)[random_index])
        new_population.pop(str(chromosome)) 
    
    new_population = dict(sorted(new_population.items(), key=operator.itemgetter(1), reverse=True))  # sorting new_population
    return new_population


# Function to print best chromosome(s) in a sorted population
def print_best_chromosomes(population): 
    first_chromosome = literal_eval(list(population)[0])
    best_fitness = population[str(first_chromosome)][1]
    for chrom in population: 
        if population[str(chrom)][1] == best_fitness: 
            print(chrom,population[str(chrom)][1],population[str(chrom)][2])


# GA Function
def GA():
    old_population = New_Population()  # creating initial sorted population
    print("Initial Population Best Chromosome(s)")
    print_best_chromosomes(old_population)
    for i in range(number_of_generations):
        new_population = nextpop(old_population)
        old_population = new_population
        print("Generation",i+1,"Best Chromosome(s)")
        print_best_chromosomes(old_population)
        
    
    
#----------------------------------------------------------------------------------   
for i in range(5):
    # dictionary keeping record of all solutions arrived at throughout the GA run i
    print("Run",i+1,"---------------------------------------------------------")
    all_chromosomes={}
    t_start = time.time()  # starting timer
    GA()
    t_end = time.time()    # stopping timer
    print("Total Time Run",i+1,"=", (t_end-t_start),"-------------------------")
    print()
      

        
        