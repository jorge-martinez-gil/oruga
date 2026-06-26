import numpy as np
from readability import Readability
from nltk.corpus import wordnet
import gensim.downloader as api
import gensim.downloader
import time
start_time = time.time()  # Record the starting time

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


# Multi-objective function to be minimized
def multi_objective(x):

    source = text
    target = obtain_text(x)

    return len([1 for i in x if i >= 1]), fitness_func1(x), float (model.wmdistance(source, target))

def jaya_multi_objective(pop_size, num_iterations, num_variables, lower_bound, upper_bound):
    population = np.random.uniform(lower_bound, upper_bound, (pop_size, num_variables))
    
    for _ in range(num_iterations):
        new_population = population.copy()
        
        for i in range(pop_size):
            for j in range(num_variables):
                rand_idx = np.random.randint(pop_size)
                new_value = population[i, j] + np.random.uniform(-1, 1) * (population[rand_idx, j] - population[i, j])
                new_value = np.clip(new_value, lower_bound, upper_bound)
                new_population[i, j] = new_value
        
        population = new_population
        
    # Calculate the objective values for each individual in the final population
    objective_values = np.array([multi_objective(individual) for individual in population])
    return population, objective_values

pop_size = 20
num_iterations = 50
num_variables = len(index_array)
lower_bound = -4
upper_bound = 4

final_population, final_objective_values = jaya_multi_objective(pop_size, num_iterations, num_variables, lower_bound, upper_bound)

print("Final Population:")
print(final_population)

print("\nFinal Objective Values:")
print(final_objective_values)

end_time = time.time()  # Record the ending time
elapsed_time = end_time - start_time  # Calculate the elapsed time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

ref_point = [60.0, 20.0, 1.0]
hypervolume = calculate_hypervolume(final_objective_values, ref_point)
print("Hypervolume:", hypervolume)