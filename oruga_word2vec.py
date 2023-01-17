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
import gensim
import gensim.downloader
import language_tool_python 
from readability import Readability
google_news_vectors = gensim.downloader.load('word2vec-google-news-300')

text_array = []
index_array = []

#text
text = 'Austria emerged from the remnants of the Eastern and Hungarian March at the end of the first millennium. Originally a margraviate of Bavaria, it developed into a duchy of the Holy Roman Empire in 1156 and was later made an archduchy in 1453. In the 16th century, Vienna began serving as the empire administrative capital and Austria thus became the heartland of the Habsburg monarchy. After the dissolution of the Holy Roman Empire in 1806, Austria established its own empire, which became a great power and the dominant member of the German Confederation. The defeat in the Austro-Prussian War of 1866 led to the end of the Confederation and paved the way for the establishment of Austria-Hungary a year later.'

r = Readability(text)
initial_score = r.flesch_kincaid().score

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

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += str(ele)
        str1 += " "
    
    str1 = str1.replace(' ,', ',')
    str1 = str1.replace('_', ' ')
    return str1
    
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
        
def obtain_text (solution): 
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
    return result
        
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
        
    if (not i[0].isupper() and len(i) > 3 and i[-2:] != 'ed'):
        number, word = Synonym(i,6)
        text_array.append (word)
        index_array.append (number)
    else:
        text_array.append (i)
        index_array.append (0)
        
    if flag == 1:
        cad = str(text_array[-1])
        text_array.pop()
        cad = cad + str(',')
        text_array.append (cad)
        flag = 0
    if flag == 2:
        cad = str(text_array[-1])
        text_array.pop()
        cad = cad + str('.')
        text_array.append (cad)
        flag = 0
        
newText = listToString(text_array)
print(newText)
print(index_array)

function_inputs = index_array
desired_output = 99
num_generations = 100 # Number of generations.
num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.
sol_per_pop = 20 # Number of solutions in the population.
num_genes = len(function_inputs)
last_fitness = 0

def on_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       fitness_func=fitness_func,
                       on_generation=on_generation)

# Running the GA to optimize the parameters of the function.
ga_instance.run()
ga_instance.plot_fitness(title="Readability evolution", ylabel="-Fitness (minimization)")

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

new_text = correct_mistakes(obtain_text(solution))
rr = Readability(new_text)
print (new_text)
print ("Difference " + str(initial_score - rr.flesch_kincaid().score))