from random import randint
from random import random
import time

class items:
	def __init__(self):
		self.values = []
		self.sizes = []
		self.weights = []
		self.length = 0

	def addItem(self, value, size, weight):
		self.values.append(value)
		self.sizes.append(size)
		self.weights.append(weight)
		self.length += 1



class knapsack:
	def __init__(self, capacity, maxWeight):
		self.capacity = capacity
		self.maxWeight = maxWeight



def createItemSet():
	itemSet = items()
	itemSet.addItem(5,2,3.5)
	itemSet.addItem(8,4,9.5)
	itemSet.addItem(6,4,4)
	itemSet.addItem(10,7,6)
	itemSet.addItem(4,3,9)
	itemSet.addItem(2,4,7.5)
	itemSet.addItem(12,8.5,5.5)
	itemSet.addItem(6,5.5,2.5)
	itemSet.addItem(6,7,9)
	itemSet.addItem(3,2,3.5)
	itemSet.addItem(5,2.5,3.5)
	itemSet.addItem(7,4,5.5)
	itemSet.addItem(9,4.5,9.5)
	itemSet.addItem(4,3,8)
	itemSet.addItem(6,5,1)
	itemSet.addItem(2,3.5,5.5)
	itemSet.addItem(6,6.5,2.5)
	itemSet.addItem(11,7.5,7)
	itemSet.addItem(7,3.5,5)
	itemSet.addItem(8,3.5,4.5)
	itemSet.addItem(3,7,2.5)
	itemSet.addItem(14,10,9)
	itemSet.addItem(4,5.5,1)
	itemSet.addItem(7,6,3)
	itemSet.addItem(6,4,3)
	return itemSet



def createPopulation(popSize, length):
	population = []
	for i in range(popSize):
		population.append(randint(0, 2**length - 1))
	return population



def fitnessFunction(chromosome, itemSet, sack):
	fitness = 0
	space = 0
	weight = 0
	# Look at each gene in the chromosome and if the allele is 1,
	# increment the counters accordingly
	for i in range(0, itemSet.length):
		if(chromosome >> itemSet.length-i-1 & 1):
			fitness += itemSet.values[i]
			space += itemSet.sizes[i]
			weight += itemSet.weights[i]
			if(space > sack.capacity or weight > sack.maxWeight):
				return 0
	return fitness

def getFitnesses(population, itemSet, sack):
	fitnesses = []
	# Calculate the fitness for every chromosome in the population
	for chromosome in population:
		fitnesses.append(fitnessFunction(chromosome, itemSet, sack))
	return fitnesses



def createCumulativeRank(fitnesses):
	total = sum(fitnesses)
	length = len(fitnesses)
	rank = [0]
	for i in range(0,length):
		rank.append(fitnesses[i]+1 + rank[i])
	return rank[1:]

def selectParent(ranking):
	val = randint(0,ranking[-1])
	i = 0
	while (val > ranking[i]):
		i += 1
	return i

def createParentsList(popSize,ranking):
	parentsList = []
	for i in range(0, popSize * 2):
		parentsList.append(selectParent(ranking))
	return parentsList



def createChild(parent1, parent2, point1, point2, length):
	part1 = (parent1 >> (length-point1)) << (length-point1)
	part2 = ((parent2 >> (length-point2)) & (2**(point2-point1) - 1)) << (length-point2)
	part3 = parent1 & (2**(length-point2) - 1)
	child = part1 + part2 + part3
	return child 

def twoPointCrossOver(parent1, parent2, pc, length):
	child1 = parent1
	child2 = parent2
	if(random() < pc):
		#print("CROSSOVER")
		i = randint(0, length)
		j = randint(0, length)
		point1 = 0
		point2 = 0
		if (i > j):
			point1 = j
			point2 = i
		else:
			point1 = i
			point2 = j
		child1 = createChild(parent1, parent2, point1, point2, length)
		child2 = createChild(parent2, parent1, point1, point2, length)
	return(child1, child2)



def mutate(chromosome, pm, length):
	mutated = chromosome
	for i in range(0,length):
		if(random() < pm):
			#print("MUTATE")
			newBit = (mutated >> (length-i-1)) & 1
			if (newBit == 1):
				newBit = 0
			else:
				newBit = 1 << (length-i-1)
			head = (mutated >> (length-i)) << (length-i)
			tail = mutated & (2**(length-i-1) - 1)
			mutated = head + newBit + tail
	return mutated



def printChromosome(chromosome):
	print("{0:b}".format(chromosome))



def startGeneticAlgorithm(popSize, maxSameFitness, pc, pm):
	start = time.time()
	itemSet = createItemSet()
	sack = knapsack(35,40)
	improvementCounter = 0
	maxFitness = -1
	fitnesses = []
	gens = 0

	# Create initial population
	population = createPopulation(popSize, itemSet.length)

	while True:
		# Get fitnesses of population
		fitnesses = getFitnesses(population, itemSet, sack)
		newMaxFitness = max(fitnesses)
		if (newMaxFitness > maxFitness):
			#print("NEW MAX FITNESS!")
			improvementCounter = 0
			maxFitness = newMaxFitness
		else:
			improvementCounter += 1

		if (improvementCounter >= maxSameFitness):
			break

		ranking = createCumulativeRank(fitnesses)
		parentsList = createParentsList(popSize, ranking)

		newPop = []

		for i in range(0,popSize):
			(child1, child2) = twoPointCrossOver(parentsList[2*i], parentsList[2*i+1], pc, itemSet.length)
			newPop.append(mutate(child1, pm, itemSet.length))
			newPop.append(mutate(child2, pm, itemSet.length))

		population = newPop
		gens += 1

	end = time.time()
	print("Terminated after {0} generations with max fitness {1} in {2}s".format(gens, max(fitnesses), end-start))
	return max(fitnesses)


def runGA(popSize, maxSameFitness, pc, pm, times):
	fitSum = 0
	for i in range(times):
		fitSum += startGeneticAlgorithm(popSize, maxSameFitness, pc, pm)
	print("Average fitness: {0}".format(fitSum/times))


def bruteForce():
	start = time.time()

	itemSet = createItemSet()
	sack = knapsack(35,40)
	numSolutions = 2**itemSet.length
	maxFitness = -1
	bestSolution = -1

	for i in range(numSolutions):
		fitness = fitnessFunction(i, itemSet, sack)
		if (fitness > maxFitness):
			maxFitness = fitness
			bestSolution = i

	print("Best solution: " + str(bestSolution) + ", fitness: " + str(maxFitness))
	printChromosome(bestSolution)

	end = time.time()

	print("Time taken: " + str(end-start))




