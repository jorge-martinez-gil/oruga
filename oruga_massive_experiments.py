# -*- coding: utf-8 -*-
"""
ORUGA: Optimizing Readability Using Genetic Algorithms

[Martinez-Gil2023a] J. Martinez-Gil, "Optimizing Readability Using Genetic Algorithms", arXiv preprint arXiv:2301.00374, 2023

@author: Jorge Martinez-Gil
"""

import csv
import pygad
import language_tool_python  
from readability import Readability
from nltk.corpus import wordnet

def main():

    text_array = []
    index_array = []
    text = ""
    global last_fitness

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
        
    def on_generation(ga_instance):
        
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
        print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
        ast_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

    with open('texts.txt', 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            text = ''.join(row)
            print (text)
            r = Readability(text)
            initial_score = r.flesch_kincaid().score
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
            #print(newText)
            print(index_array)

            function_inputs = index_array
            desired_output = 99
            num_generations = 100 # Number of generations.
            num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.
            sol_per_pop = 20 # Number of solutions in the population.
            num_genes = len(function_inputs)
            last_fitness = 0



            ga_instance = pygad.GA(num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                sol_per_pop=sol_per_pop,
                                num_genes=num_genes,
                                fitness_func=fitness_func,
                                on_generation=on_generation)

            # Running the GA to optimize the parameters of the function.
            ga_instance.run()
            #ga_instance.plot_fitness(title="Readability evolution", ylabel="-Fitness (minimization)")

            # Returning the details of the best solution.
            solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
            print("Parameters of the best solution : {solution}".format(solution=solution))
            print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
            print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

            new_text = correct_mistakes(obtain_text(solution))
            rr = Readability(new_text)

            with open('results.txt', 'a') as the_file:
                the_file.write("Difference " + str(initial_score - rr.flesch_kincaid().score) + str('\n'))



if __name__ == "__main__":
    for x in range(10):
        main()