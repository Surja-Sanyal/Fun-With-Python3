#-------------------------------------------------------------------------------------------------------------------------------------------------------------:
#Problem:		TSP - Using GA
#Inputs:		Number of cities in TSP
#			    Space separated cost matrix of inter-city travel
#Outputs:		An optimal sequence of cities to travel for the Salesman
#Created By:	Surja Sanyal
#Created On:	03 FEB 2020
#Roll Number:	19CS4105
#Contact:		8595320732
#Mail To:		hi.surja06@gmail.com
#Modified On:	04 MAY 2020
#-------------------------------------------------------------------------------------------------------------------------------------------------------------:



###	Setup	###

import os
import sys
import math
import tsplib95
import numpy as np
from random import randint



###	Function Definitions	###


##	Input TSP details	##

#Choose predefined problem
def choose_predefined_problem():

    #User input predefined problem number or new problem option
    print ("Please choose a predefined problem (1 through ", len(list(os.walk(sys.path[0] + "/ALL_atsp/"))) - 1, ") to work with or choose to enter a new TSP problem (any other option): ", sep="", end="")
    return int(input()) - 1


#Get problem dimension from predefined problem
def get_problem_dimension(problem_number):
    
    #Get problem dimension from predefined problem
    problem_location = sys.path[0] + "/ALL_atsp/" 
    problem_folder = sorted(os.listdir(problem_location))[problem_number]
    problem = tsplib95.load_problem(problem_location + problem_folder + "/" + problem_folder)
    return problem.dimension


#Assign adjacency matrix from predefined TSP problem
def assign_adjacency(problem):
    
    #Get adjacency from predefined problem
    for i in range(n):
        for j in range(n):
            adjacency[i, j] = problem.wfunc(i, j)


#Assign predefined problem
def assign_problem(problem_number, adjacency):
    
    #Assign predefined problem
    problem_location = sys.path[0] + "/ALL_atsp/" 
    problem_folder = sorted(os.listdir(problem_location))[problem_number]
    problem = tsplib95.load_problem(problem_location + problem_folder + "/" + problem_folder)	
    assign_adjacency(problem)


#Input TSP cities
def input_cities():

	#User input number of cities in the TSP
	print ("Please enter the number of cities: ", end="")
	return int(input())


#Input travel costs
def input_adjacency():

	#User input the cost matrix for the inter-city travels of the TSP
	print ("Please enter the adjacency matrix (space separated): ", end="")
	input_line_buffer=input().split()

	#Split space separated cost matrix into a 2-D numpy array
	for i in range(n):
		for j in range(n):
			adjacency[i, j]=int(input_line_buffer[i * n + j])



##	Initialize Population	##

#Round to even numbers for simplicity
def even_round(any_number):
	if (math.ceil(any_number) % 2 == 0):
		return (math.ceil(any_number))
	else:
		return (math.ceil(any_number) + 1)


#Calculate how many solutions to take in the population
def calculate_population():
	return even_round(max(math.floor(math.log2(n)), 1) * 4)


#Convert an array to a single chromosome
def convert_to_single():
    s = [str(i).zfill(3) for i in chromosome]
    return "".join(s)


#Choose the solutions for the initial population and encode them as chromosomes
def initialize_population(m):

	#Track distinct chromosomes
	count_distinct_chromosomes = 0
	while (count_distinct_chromosomes < m):
		chromosome_repeated_flag = 0

		#Create a random chromosome
		np.random.shuffle(chromosome)

		#Encode to an integer chromosome
		single_chromosome = convert_to_single()

		#Check for duplicacy
		for j in range(count_distinct_chromosomes):
			if (int(single_chromosome) == population[j]):
				chromosome_repeated_flag = 1
				break

		#Include in population if not duplicate
		if (chromosome_repeated_flag == 0):
			population[count_distinct_chromosomes] = single_chromosome
			count_distinct_chromosomes += 1



##	Evaluate Fitness	##

def evaluate_fitness():

	#Track minimum fitness achieved so far
	global current_minimum_fitness

	#Evaluate fitness for each chromosome
	for i in range(m):
		total_fitness = 0
		current_chromosome = str(population[i])

		#Calculate costs for each inter-city travel
		for j in range(n - 1):
			from_city = current_chromosome[3 * j:3 * (j + 1)]
			to_city = current_chromosome[3 * (j + 1):3 * (j + 2)]
			total_fitness += adjacency[int(from_city) - 1, int(to_city) - 1]
		total_fitness += adjacency[int(current_chromosome[3 * n - 3:3 * n]) - 1, int(current_chromosome[0:0 + 3]) - 1]

		#Merge all inter-city travel costss into one tour cost
		fitness[i] = total_fitness

	#Update minimum fitness achieved so far
	if (min(fitness) < current_minimum_fitness):
		current_minimum_fitness = min(fitness)



##	Selection	##

def selection():

	#select m chromosomes randomly
	count_selected_chromosomes = 0
	while (count_selected_chromosomes < m):

		#Random index for selection
		next_random_index = randint(0, m - 1)

		#Selection proportional to fitness
		count_selected_frequency = min(math.ceil(((max(max(fitness) - fitness[next_random_index], 1))/max((max(fitness) - min(fitness)), 1)) * (m/5)), m - count_selected_chromosomes)

		#Put selected chromosomes in new population
		for i in range(count_selected_frequency):
			new_population[count_selected_chromosomes + i] = population[next_random_index]
		count_selected_chromosomes += count_selected_frequency



##	Crossover	##

def crossover():

	#Crossover done in pairs
	for i in range(0, m, 2):

		#Crossover probability 0.8
		if (randint(1, 100) <= 80):

			#Find random parents
			first_parent_index = second_parent_index = -1
			while (crossover_recorder[first_parent_index] == 1 or first_parent_index < 0):
				first_parent_index = randint(0, m - 1)
			while (crossover_recorder[second_parent_index] == 1 or second_parent_index == first_parent_index or second_parent_index < 0):
				second_parent_index = randint(0, m - 1)
			crossover_recorder[first_parent_index] = crossover_recorder[second_parent_index] = 1

			#Find random crossover points
			first_crossover_point = randint(1, math.floor(n/2))
			second_crossover_point = randint(math.ceil(n/2), n - 2)

			#Order 1 crossover
			first_child = str(population[first_parent_index])[3 * first_crossover_point:3 * second_crossover_point + 3]
			second_child = str(population[second_parent_index])[3 * first_crossover_point:3 * second_crossover_point + 3]

			first_parent_genes = second_parent_genes = ""
			for i in range(n):
				got_match = 0
				find_char = str(population[second_parent_index])[3 * i:3 * i + 3]
				for j in range(n):
					current_gene = first_child[3 * j:3 * j + 3]
					if (current_gene == find_char):
						got_match = 1
						break
				if (got_match == 0):
					second_parent_genes += find_char

			for i in range(n):
				got_match = 0
				find_char = str(population[first_parent_index])[3 * i:3 * i + 3]
				for j in range(n):
					current_gene = second_child[3 * j:3 * j + 3]
					if (current_gene == find_char):
						got_match = 1
						break
				if (got_match == 0):
					first_parent_genes += find_char

			#Put children into population in place of parents
			population[first_parent_index] = second_parent_genes[0:3 * first_crossover_point] + first_child[0:] + second_parent_genes[3 * first_crossover_point:]
			population[second_parent_index] = first_parent_genes[0:3 * first_crossover_point] + second_child[0:] + first_parent_genes[3 * first_crossover_point:]




##	Mutation	##

def mutation():

	#For all chromosomes
	for i in range(m):

		#Mutation probability 0.08
		if (randint(1, 100) <= 8):

			#Get swap locations
			first_swap_location = second_swap_location = randint(0, n - 1)
			while (first_swap_location == second_swap_location):
				second_swap_location = randint(0, n - 1)

			#Mutate chromosomes
			mutated_chromosome = list(map(''.join, zip( * [iter(population[i])] * 3)))
			mutated_chromosome[first_swap_location], mutated_chromosome[second_swap_location] = mutated_chromosome[second_swap_location], mutated_chromosome[first_swap_location]

			#put back mutated chromosomes into population
			population[i] = "".join(mutated_chromosome)



##	 Print Input Cost Matrix	##

def print_problem():

	print ("\n\n\n\n--The Problem Details--")

	#print number of cities in TSP
	print ("\nInput Cost Matrix: For", n, "Cities -\n\n")

	#print cost matrix
	print ("\t", end="")
	for i in range(n):
		print ("City", i + 1, end="\t")
	print ("")

	for i in range(n):
		print ("City", i + 1, end="\t")
		for j in range(n):
			print (adjacency[i, j], "\t", end="")
		print ("")



##	Print Optimal Tour Details	##

def print_tour():

	print ("\n\n--The Solution Details--")

	#Print optimal cost
	print ("\nCurrent minimum cost TSP Tour having cost", min(fitness), "obtained in", iteration_count, "iterations is:\n")

	#Print optimal route
	route = list(map(''.join, zip( * [iter(str(population[np.where(fitness == min(fitness))[0][0]]))] * 3)))#list(str(population[np.where(fitness == min(fitness))[0][0]]))
	for i in range(n):
		print (int(route[i]), "--> ", end="")
	print (int(route[0]), ".\n", sep="")




####	Main Body	####


##	Clear Terminal	##

os.system("clear")



##	Choose/Input TSP problem details	##

#   Choose predefined problems (1 through 10) or enter new problem   #

predefined_problems = np.arange(len(list(os.walk(sys.path[0] + "/ALL_atsp/"))) - 1)
q = choose_predefined_problem()


#   Check for choice option   #

if (q not in predefined_problems):
    
    #	Input number of cities	#
    
    n = input_cities()
    
    
    #	Input cost of inter-city travels	#
    
    adjacency = np.zeros((n, n), dtype=np.int32)
    input_adjacency()


else:
    
    #Get predefined problem details
    
    n = get_problem_dimension(q)
    adjacency = np.zeros((n, n), dtype=np.int32)
    assign_problem(q, adjacency)


#Print input cost matrix
print_problem()
print ("\n\n\n\n")



##	Initialize Population	##

m = calculate_population()
chromosome = np.arange(n) + 1
population = [""] * m
initialize_population(m)
print ("Population:\n", population)



##	Fitness Evaluation	##

current_minimum_fitness = sum(sum(adjacency))
fitness = np.zeros(m, dtype=np.int32)
evaluate_fitness()
print ("Fitness:\n", fitness)



##	Terminate?	##

#Termination by fitness achieved
iteration_count = 0

while (max(fitness) != min(fitness) or min(fitness) > current_minimum_fitness):
#while (max(fitness) != min(fitness)):



    ##	Selection	##

    new_population = [''] * m
    selection()
    population = new_population
    print ("Selection:\n", population)



    ##	Crossover	##

    crossover_recorder = np.zeros(m, dtype=np.int32)
    crossover()
    print ("Crossover:\n", population)



    ##	Mutation	##

    mutation()
    print ("Mutation:\n", population)



    ##	Fitness Evaluation	##

    fitness = np.zeros(m, dtype=np.int32)
    evaluate_fitness()
    print ("Fitness:\n", fitness)
    print ("Best fitness achieved in any iteration is:", current_minimum_fitness)



    ##	Track Iterations	##

    iteration_count += 1




##	Output Solution	##

#Print input cost matrix
print_problem()

#Print minimum TSP Tour details
print_tour()



##	End of Code	##



