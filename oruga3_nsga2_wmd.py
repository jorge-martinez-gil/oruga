# -*- coding: utf-8 -*-
"""
ORUGA: Optimizing Readability Using Genetic Algorithms

[Martinez-Gil2023a] J. Martinez-Gil, "Optimizing Readability Using Genetic Algorithms", arXiv preprint arXiv:2301.00374, 2023

@author: Jorge Martinez-Gil
"""

from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from readability import Readability
from nltk.corpus import wordnet
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
import gensim.downloader as api
model = api.load('word2vec-google-news-300')


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
        if solution[index] <= 0:
            text_converted.append (i)
        elif solution[index] > 0:
            number, word = Synonym(i,solution[index])
            text_converted.append (word)
        else: 
            print ('Error')
        index += 1
        
    result = listToString(text_converted)
    r = Readability(result)
    return r.flesch_kincaid().score

text = 'Mapagala fortress was an ancient fortified complex of the Anuradhapura Kingdom long before Kasyapa I built his city, Sigiriya. It is located to the South of Sigiriya and closer to Sigiriya tank. It was built by using unshaped boulders to about 20 ft high. Each stone is broad and thick and some of them are about 10 ft high and about 4 ft long. It is believed that it was built before the time of usage of metal tools. Arthur Maurice Hocart noted that cyclopean style stone walls were used for the fortress, and square hammered stones were used for the ramparts of the citadel. However, his note suggests metal (iron) tools were used for construction. Excavations work in this areas found a few stone forges, which proved the claim on the usage of metal tools.'
    
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
        if solution[index] <= 0:
            text_converted.append (i)
        elif solution[index] > 0:
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

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        source = text
        target = obtain_text(solution.variables)

        solution.objectives[2] = float (model.wmdistance(source, target))
        solution.objectives[1] = fitness_func1(solution.variables)
        solution.objectives[0] = len([1 for i in solution.variables if i > 0])

        return solution


    def get_name(self):
        return 'Oruga'

problem = Oruga()
algorithm = NSGAII(
    problem=problem,
    population_size=20,
    offspring_population_size=30,
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations=800)
)

algorithm.run()

from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.lab.visualization import Plot

front = get_non_dominated_solutions(algorithm.get_result())


# save to files
print_function_values_to_file(front, 'FUN.NSGAII')
print_variables_to_file(front, 'VAR.NSGAII')
plot_front = Plot(title='ORUGA', axis_labels=['Words to be replaced', 'Readability Score', 'Semantic Distance'])
plot_front.plot(front, label='NSGA-II', filename='NSGAII-ORUGA', format='png')

for solution in front:
    print (obtain_text(solution.variables))