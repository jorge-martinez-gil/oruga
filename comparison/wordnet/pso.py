# -*- coding: utf-8 -*-
"""
ORUGA: Optimizing Readability Using Particle Swarm Optimization (PSO)

[Martinez-Gil2023a] J. Martinez-Gil, "Optimizing Readability Using Genetic Algorithms", arXiv preprint arXiv:2301.00374, 2023

@author: Jorge Martinez-Gil
"""

from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator import PolynomialMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.solution import get_non_dominated_solutions
from nltk.corpus import wordnet
from readability import Readability
import gensim.downloader as api

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
    # preprocessing
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
    return r.flesch_kincaid().score

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


class Oruga(FloatProblem):

    def __init__(self):
        super(Oruga, self).__init__()
        self.number_of_objectives = 3
        self.number_of_variables = len(index_array)
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        source = text
        target = obtain_text(solution.variables)

        solution.objectives[2] = float(model.wmdistance(source, target))
        solution.objectives[1] = fitness_func1(solution.variables)
        solution.objectives[0] = len([1 for i in solution.variables if i >= 1])

        return solution

    def get_name(self):
        return 'Oruga'

problem = Oruga()

max_evaluations = 100
algorithm = SMPSO(
    problem=problem,
    swarm_size=100,
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
    leaders=CrowdingDistanceArchive(100),
    termination_criterion=StoppingByEvaluations(max_evaluations)
)

algorithm.run()
solutions = algorithm.get_result()
front = get_non_dominated_solutions(solutions)

# Print number of solutions
print (len(front))

# Print time
print (algorithm.total_computing_time)

# Define the reference point (maximum values for each objective)
ref_point = [30.0, 20.0, 1.0]

pareto_front = [[solution.objectives[0], solution.objectives[1], solution.objectives[2]] for solution in front]

print("Pareto Front:")
for point in pareto_front:
    print(point)

# Calculate the hypervolume
hypervolume = calculate_hypervolume(pareto_front, ref_point)
print("Hypervolume:", hypervolume)








