#-------------------------------------------------------------------------------------------------------------------------------------------------------------:
#Problem:       TSP - Using GA
#Inputs:        Number of cities in TSP
#				Space separated cost matrix of inter-city travel
#Outputs:       An optimal sequence of cities to travel for the Salesman
#Created By:    Surja Sanyal
#Created On:    03 FEB 2020
#Roll Number:   19CS4105
#Contact:		8595320732
#Mail To:       hi.surja06@gmail.com
#Modified On:   NA
#-------------------------------------------------------------------------------------------------------------------------------------------------------------:



###	Setup	###

import os
import math
import numpy as np
from random import randint




###	Sample problem and solution for trial	###
###	Please Comment code as guided in the "Main Body" --> "Input TSP problem details" section	###

#Sample number of cities for below cost matrix: 7
#Sample cost matrix for user input: 45 23 65 33 76 98 43 23 56 78 98 65 43 12 34 56 78 65 43 23 65 76 87 37 54 73 85 63 27 48 65 49 65 84 63 74 56 38 64 65 43 12 34 56 78 65 49 65 84 63 74
#Optimal Solution to above sample: Cost:- 225, Route:- 7 --> 5 --> 2 --> 1 --> 4 --> 3 --> 6 --> 7.





###	Function Definitions	###


##	Input TSP details	##

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


#Calculate how many nsolutions to take in the population
def calculate_population():
	return even_round(math.log10(n) * 20)


#Convert a string chromosome to an integer encoding
def convert_to_int():
	return int("".join(map(str, chromosome)))


#Choose the solutions for the initial population and encode them as chromosomes
def initialize_population(m):

	#Track distinct chromosomes
	count_distinct_chromosomes = 0
	while (count_distinct_chromosomes < m):
		chromosome_repeated_flag = 0

		#Create a random chromosome
		np.random.shuffle(chromosome)

		#Encode to an integer chromosome
		numeric_chromosome = convert_to_int()

		#Check for duplicacy
		for j in range(count_distinct_chromosomes):
			if (numeric_chromosome == population[j]):
				chromosome_repeated_flag = 1
				break

		#Include in population if not duplicate
		if (chromosome_repeated_flag == 0):
			population[count_distinct_chromosomes] = numeric_chromosome
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
			from_city = current_chromosome[j:j + 1]
			to_city = current_chromosome[j + 1:j + 2]
			total_fitness += adjacency[int(from_city) - 1, int(to_city) - 1]
		total_fitness += adjacency[int(current_chromosome[n - 1:n]) - 1, int(current_chromosome[0:0 + 1]) - 1]

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
		count_selected_frequency = min(math.ceil(((max(fitness) - fitness[next_random_index] + 1)/max(fitness)) * (m/2)), m - count_selected_chromosomes)

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
			first_child = str(population[first_parent_index])[first_crossover_point:second_crossover_point]
			second_child = str(population[second_parent_index])[first_crossover_point:second_crossover_point]

			first_parent_genes = second_parent_genes = ""
			for i in range(n):
				find_char = str(population[second_parent_index])[i:i + 1]
				if (first_child.find(find_char) == -1):
					second_parent_genes += find_char

			for i in range(n):
				find_char = str(population[first_parent_index])[i:i + 1]
				if (second_child.find(find_char) == -1):
					first_parent_genes += find_char

			first_child = second_parent_genes[0:first_crossover_point] + first_child + second_parent_genes[first_crossover_point:]
			second_child = first_parent_genes[0:first_crossover_point] + second_child + first_parent_genes[first_crossover_point:]

			#Put children into population in place of parents
			population[first_parent_index] = int(first_child)
			population[second_parent_index] = int(second_child)



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
			mutated_chromosome = list(str(population[i]))
			mutated_chromosome[first_swap_location], mutated_chromosome[second_swap_location] = mutated_chromosome[second_swap_location], mutated_chromosome[first_swap_location]

			#put back mutated chromosomes into population
			population[i] = int("".join(mutated_chromosome))



##	 Print Input Cost Matrix	##

def print_cost():

	print ("\n\n\n\n--The Problem Details--")

	#Print number of cities in TSP
	print ("\nInput Cost Matrix: For", n, "Cities:\n\n")

	#Print cost matrix
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
	route = list(str(population[np.where(fitness == min(fitness))[0][0]]))
	for i in range(n):
		print (route[i], "--> ", end="")
	print (route[0], ".\n", sep="")




####	Main Body	####


##	Clear Terminal	##

os.system("clear")



##	Input TSP problem details	##

#	Input number of cities	#

n = input_cities()	#Uncomment for User Input
#n = 5	#Comment for User Input


#	Input cost of inter-city travels	#

adjacency = np.zeros((n, n), dtype=np.int32)
#adjacency = np.array([(0, 2, 0, 6, 1), (1, 0, 4, 4, 2), (5, 3, 0, 1, 5), (4, 7, 2, 0, 1), (2, 6, 3, 6, 0)])	#Comment for User Input
input_adjacency()	#Uncomment for User Input


#	Sample problem and solution for trial	#
#	Please Comment code as guided in the above sectionm section	#

#Sample number of cities for below cost matrix: 7
#Sample cost matrix for user input: 45 23 65 33 76 98 43 23 56 78 98 65 43 12 34 56 78 65 43 23 65 76 87 37 54 73 85 63 27 48 65 49 65 84 63 74 56 38 64 65 43 12 34 56 78 65 49 65 84 63 74
#Optimal Solution to above sample: Cost:- 225, Route:- 7 --> 5 --> 2 --> 1 --> 4 --> 3 --> 6 --> 7.



##	Initialize Population	##

m = calculate_population()
chromosome = np.arange(n) + 1
population = np.zeros(m, dtype=np.int32)
initialize_population(m)
print ("Population:", population, sep="\t")



##	Fitness Evaluation	##

current_minimum_fitness = sum(sum(adjacency))
fitness = np.zeros(m, dtype=np.int32)
evaluate_fitness()
print ("Fitness:", fitness, sep="\t")



##	Terminate?	##

#Termination by iterations or by fitness achieved
iteration_count = 0
while (max(fitness) != min(fitness) or min(fitness) > current_minimum_fitness):



    ##	Selection	##

    new_population = np.zeros(m, dtype=np.int32)
    selection()
    population = new_population
    print ("Selection:", population, sep="\t")



    ##	Crossover	##

    crossover_recorder = np.zeros(m, dtype=np.int32)
    crossover()
    print ("Crossover:", population, sep="\t")



    ##	Mutation	##

    mutation()
    print ("Mutation:", population, sep="\t")



    ##	Fitness Evaluation	##

    fitness = np.zeros(m, dtype=np.int32)
    evaluate_fitness()
    print ("Fitness.", fitness, sep="\t")



    ##	Track Iterations	##

    iteration_count += 1




##	Output Solution	##

#Print input cost matrix
print_cost()

#Print minimum TSP Tour details
print_tour()



##	End of Code	##



