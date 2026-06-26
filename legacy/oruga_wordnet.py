# -*- coding: utf-8 -*-
"""
ORUGA: Optimizing Readability Using Genetic Algorithms

[Martinez-Gil2023a] J. Martinez-Gil, "Optimizing Readability Using Genetic Algorithms", arXiv preprint arXiv:2301.00374, 2023

@author: Jorge Martinez-Gil
"""

#print(r.flesch_kincaid().score)
#print(r.flesch().score)
#print(r.gunning_fog())
#print(r.coleman_liau())
#print(r.dale_chall())
#print(r.ari())
#print(r.linsear_write())
#print(r.spache())

#Coding of individuals
#-2, candidate but not synonym
#-1, special character (if necessary)
#0, not candidate
#1, replaced by 1st option
#2, replaced by 2nd option
#N, replaced by Nth option

# Modules
import pygad
import language_tool_python  
from readability import Readability
from nltk.corpus import wordnet

text_array = []
index_array = []

#Text
text = 'Niagara Falls is a group of three waterfalls at the southern end of Niagara Gorge, spanning the border between the province of Ontario in Canada and the state of New York in the United States. The largest of the three is Horseshoe Falls, which straddles the international border of the two countries. It is also known as the Canadian Falls. The smaller American Falls and Bridal Veil Falls lie within the United States. Bridal Veil Falls is separated from Horseshoe Falls by Goat Island and from American Falls by Luna Island, with both islands situated in New York. Formed by the Niagara River, which drains Lake Erie into Lake Ontario, the combined falls have the highest flow rate of any waterfall in North America that has a vertical drop of more than 50 m (160 ft). During peak daytime tourist hours, more than 168,000 m3 (5.9 million cu ft) of water goes over the crest of the falls every minute.'

r = Readability(text)
initial_score = r.flesch_kincaid().score

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
    
def correct_mistakes (text):
    my_tool = language_tool_python.LanguageTool('en-US')  
    my_text = text  
    my_matches = my_tool.check(my_text)  
  
    myMistakes = []  
    myCorrections = []  
    startPositions = []  
    endPositions = []  
  
    # using the for-loop  
    for rules in my_matches:  
        if len(rules.replacements) > 0:  
            startPositions.append(rules.offset)  
            endPositions.append(rules.errorLength + rules.offset)  
            myMistakes.append(my_text[rules.offset : rules.errorLength + rules.offset])  
            myCorrections.append(rules.replacements[0])  
  
    # creating new object  
    my_NewText = list(my_text)   
  
    # rewriting the correct passage  
    for n in range(len(startPositions)):  
        for i in range(len(my_text)):  
            my_NewText[startPositions[n]] = myCorrections[n]  
            if (i > startPositions[n] and i < endPositions[n]):  
                my_NewText[i] = ""  
  
    my_NewText = "".join(my_NewText)  
    
    return my_NewText  
        
def fitness_func(solution, solution_idx):

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
    return r.flesch_kincaid().score * -1

print (text)
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
        
newText = listToString(text_array)
print(newText)
print(index_array)

# Parameters for the GA
function_inputs = index_array
num_generations = 100  # Number of generations
num_parents_mating = 10  # Number of solutions to be selected as parents in the mating pool
sol_per_pop = 20  # Number of solutions in the population
num_genes = len(function_inputs)  # Number of genes

# Initialize the GA instance without the 'on_generation' argument
ga_instance = pygad.GA(num_generations=1,  # Set to 1 because we are controlling the generations manually
                        num_parents_mating=num_parents_mating,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        fitness_func=fitness_func)

last_fitness = 0  # Initialize last fitness for comparison

# Manually iterate through generations
for generation in range(num_generations):
    ga_instance.run()  # Run GA for one generation

    # Getting the best solution after the current generation
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    print("Generation = {}".format(generation + 1))
    print("Fitness    = {}".format(solution_fitness))
    print("Change     = {}".format(solution_fitness - last_fitness))

    last_fitness = solution_fitness  # Update the last fitness value

# At this point, the GA has completed all generations
# You can directly get the best solution details without passing any arguments
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

new_text = correct_mistakes(obtain_text(solution))
rr = Readability(new_text)
print(new_text)
print("Difference " + str(initial_score - rr.flesch_kincaid().score))
