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
import gensim.downloader

model = api.load('word2vec-google-news-300')
google_news_vectors = gensim.downloader.load('word2vec-google-news-300')


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
    
    if (Dict.get(word) is not None):
        synonyms = Dict.get(word)
             
    if (not synonyms):
        return -2, word
    elif number >= len(synonyms):
        return len(synonyms)-1, synonyms[len(synonyms)-1][0]
    else:
        return int(number), synonyms[int(number-1)][0]
       
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


#Creates a dictionary in order to store all the synonyms in main memory
resource = text.split() 
Dict = {}
for i in resource:
    if ',' in i:
        i = i.replace(',', '')
    if '.' in i:
        i = i.replace('.', '') 

    if (not i[0].isupper() and len(i) > 3):
        if i in Dict.keys():
            print ("Processing...Please wait")
        else:
            try:
                synonyms = google_news_vectors.most_similar(i, topn=6)
            except KeyError as e:
                print (e)
                synonyms = None
            if synonyms is not None:
                Dict[i] = []
                Dict[i] = synonyms

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
        solution.objectives[0] = len([1 for i in solution.variables if i >= 1])

        return solution


    def get_name(self):
        return 'Oruga'

max_evaluations = 3000
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

from jmetal.util.solution import get_non_dominated_solutions

front = get_non_dominated_solutions(algorithm.get_result())

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