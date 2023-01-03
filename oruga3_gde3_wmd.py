# -*- coding: utf-8 -*-
"""
ORUGA: Optimizing Readability Using Genetic Algorithms

[Martinez-Gil2023a] J. Martinez-Gil, "Optimizing Readability Using Genetic Algorithms", arXiv preprint arXiv:2301.00374, 2023

@author: Jorge Martinez-Gil
"""

from jmetal.algorithm.multiobjective.gde3 import GDE3
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
    return r.ari().score

text = 'Big data refers to data sets that are too large or complex to be dealt with by traditional data-processing application software. Data with many fields (rows) offer greater statistical power, while data with higher complexity (more attributes or columns) may lead to a higher false discovery rate. Big data analysis challenges include capturing data, data storage, data analysis, search, sharing, transfer, visualization, querying, updating, information privacy, and data source. Big data was originally associated with three key concepts volume, variety, and velocity. The analysis of big data presents challenges in sampling, and thus previously allowing for only observations and sampling. Thus a fourth concept, veracity, refers to the quality or insightfulness of the data.'
    
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

max_evaluations = 3000
problem = Oruga()
algorithm = GDE3(
    problem=problem,
    population_size=100,
    cr=0.5,
    f=0.5,
    termination_criterion=StoppingByEvaluations(max_evaluations)
)

algorithm.run()

from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.lab.visualization import Plot

front = get_non_dominated_solutions(algorithm.get_result())


# save to files
print_function_values_to_file(front, 'FUN.GDE3')
print_variables_to_file(front, 'VAR.GDE3')
plot_front = Plot(title='ORUGA', axis_labels=['Words to be replaced', 'Readability Score'])
plot_front.plot(front, label='GDE3', filename='GDE3-ORUGA', format='png')

for solution in front:
    # We should call here a function to try to correct the text
    print (obtain_text(solution.variables))