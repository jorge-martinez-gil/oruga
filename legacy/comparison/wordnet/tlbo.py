import numpy as np
from readability import Readability
from nltk.corpus import wordnet
import random
import gensim.downloader as api
import nltk
from nltk.corpus import wordnet
from deap import base, creator, tools

model = api.load('word2vec-google-news-300')

def calculate_hypervolume(pareto_front, ref_point):
    sorted_front = sorted(pareto_front, key=lambda x: x[0])
    hypervolume = 0.0
    prev_point = [0.0, ref_point[1]]

    for point in sorted_front:
        if point[1] < prev_point[1]:
            hypervolume += (prev_point[0] - point[0]) * (prev_point[1] - ref_point[1])
            prev_point = point

    hypervolume += (prev_point[0] - ref_point[0]) * (prev_point[1] - ref_point[1])
    return hypervolume

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += str(ele)
        str1 += " "
    
    str1 = str1.replace(' ,', ',')
    str1 = str1.replace('_', ' ')
    return str1

def Synonym(word, number):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
            synonyms.append(lm.name())
             
    if (not synonyms):
        return -2, word
    elif number >= len(synonyms):
        return len(synonyms)-1, synonyms[len(synonyms)-1]
    else:
        return int(number), synonyms[int(number-1)] 
       
def fitness_func1(solution):
    print (solution)

    #preprocessing
    a = 0
    for i in index_array:
        if index_array[a] <= 0:
            solution[a] = 0
        a += 1
    
    res2 = text.split() 
    text_converted = []
    index=0
    for i in res2:
        if solution[index] < 1:
            text_converted.append (i)
        elif solution[index] >= 1:
            number, word = Synonym(i,solution[index])
            text_converted.append (word)
        else: 
            print ('Error')
        index += 1
        
    result = listToString(text_converted)
    r = Readability(result)
    return r.ari().score

text = 'The sea moderates the climate and has important roles in the water cycle, carbon cycle, and nitrogen cycle. Humans harnessing and studying the sea have been recorded since ancient times, and evidenced well into prehistory, while its modern scientific study is called oceanography. The most abundant solid dissolved in seawater is sodium chloride. The water also contains salts of magnesium, calcium, potassium, and mercury, amongst many other elements, some in minute concentrations. Salinity varies widely, being lower near the surface and the mouths of large rivers and higher in the depths of the ocean; however, the relative proportions of dissolved salts vary little across the oceans.'
    
text_array = []
index_array = []

res = text.split()
for i in res:
    flag = 0
    if ',' in i:
        i = i.replace(',', '')
        flag = 1
    if '.' in i:
        i = i.replace('.', '')
        flag = 2   
        
    if (not i[0].isupper() and len(i) > 3):
        number, word = Synonym(i,6)
        text_array.append (word)
        index_array.append (number)
    else:
        text_array.append (i)
        index_array.append (0)
        
    if flag == 1:
        cad = text_array[-1]
        text_array.pop()
        cad = cad + str(',')
        text_array.append (cad)
        flag = 0
    if flag == 2:
        cad = text_array[-1]
        text_array.pop()
        cad = cad + str('.')
        text_array.append (cad)
        flag = 0
        
def obtain_text (solution): 
    res2 = text.split() 
    text_converted = []
    index=0
    for i in res2:
        if solution[index] < 1:
            text_converted.append (i)
        elif solution[index] >= 1:
            number, word = Synonym(i,solution[index])
            text_converted.append (word.upper())
        else: 
            print ('Error')
        index += 1
        
    result = listToString(text_converted)
    return result

# Define your problem's objective function
def objective_function(x):
    
    source = text
    target = obtain_text(x)

    return [len([1 for i in x if i >= 1]), fitness_func1(x), float (model.wmdistance(source, target))]

# TLBO parameters
population_size = 20
max_generations = 50
dimension = len(index_array)
lower_bound = -4
upper_bound = 4

# DEAP initialization
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, lower_bound, upper_bound)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dimension)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", objective_function)

def teaching_phase(learners, mean_teacher):
    for learner in learners:
        diff = [mean_teacher[dim] - learner[dim] for dim in range(dimension)]
        random_values = [random.random() for _ in range(dimension)]
        update_vector = [random_value * diff[dim] for dim, random_value in enumerate(random_values)]
        learner[:] = [learner[dim] + update_vector[dim] for dim in range(dimension)]

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selBest)


def main():
    population = toolbox.population(n=population_size)

    # Attach fitness values to the initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    pareto_front = []  # Initialize Pareto front

    # TLBO main loop
    for generation in range(max_generations):
        teachers = toolbox.select(population, k=5)
        mean_teacher = [sum(teacher[dim] for teacher in teachers) / len(teachers) for dim in range(dimension)]
    
        learners = population[:]
        teaching_phase(learners, mean_teacher)
    
        offspring = learners[:]  # No genetic operations
    
        # Attach fitness values to the offspring
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
    
        for i in range(population_size):
            offspring_fitness = offspring[i].fitness.values
            for ind in population:
                if ind != offspring[i]:
                    ind_fitness = ind.fitness.values
                    is_dominated = all(offspring_fitness[dim] <= ind_fitness[dim] for dim in range(len(offspring_fitness)))
                    if not is_dominated:
                        if offspring[i] not in pareto_front:
                            pareto_front.append(offspring[i])
        
    final = []
    front = []
    for item in pareto_front:
        if item not in front:
            front.append(item)
    
    # Print fitness
    for solution in front:
        final.append(solution.fitness.values)
        print("Fitness:", solution.fitness.values)

    ref_point = [30.0, 20.0, 1.0]
    hypervolume = calculate_hypervolume(final, ref_point)
    print("Hypervolume:", hypervolume)
    
    print (len(pareto_front))

if __name__ == "__main__":
    main()

















