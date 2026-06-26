import numpy as np
from readability import Readability
from nltk.corpus import wordnet
import random
import gensim.downloader as api
model = api.load('word2vec-google-news-300')

# Your existing imports here
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

# Define the multi-objective optimization problem
def evaluate(x):
    source = text
    target = obtain_text(x)

    return len([1 for i in x if i >= 1]), fitness_func1(x), float (model.wmdistance(source, target))

def generate_random_solution(bounds):
    return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]

def levy_flight(beta):
    return np.power((1.0 / np.random.gamma(1.0 + beta)), 1.0 / beta)

def cuckoo_search_multiobjective(bounds, generations, population_size, pa):
    dim = len(bounds)
    population = [generate_random_solution(bounds) for _ in range(population_size)]
    
    for gen in range(generations):
        population.sort(key=lambda x: evaluate(x))
        new_population = population[:population_size//2]
        
        for _ in range(population_size - population_size//2):
            if random.random() < pa:
                selected_cuckoo = random.choice(new_population)
                cuckoo = [x + levy_flight(1.5) * (x - y) for x, y in zip(selected_cuckoo, population[random.randint(0, population_size//2-1)])]
                cuckoo = np.clip(cuckoo, bounds[:, 0], bounds[:, 1])
                new_population.append(cuckoo)
            else:
                new_population.append(generate_random_solution(bounds))
        
        population = new_population
    
    population.sort(key=lambda x: evaluate(x))
    
    pareto_front = []
    for ind in population:
        dominated = False
        to_remove = []
        for idx, existing in enumerate(pareto_front):
            if all(a <= b for a, b in zip(existing, ind)):
                to_remove.append(idx)
            elif all(a >= b for a, b in zip(existing, ind)):
                dominated = True
                break
        if not dominated:
            pareto_front = [existing for idx, existing in enumerate(pareto_front) if idx not in to_remove]
            pareto_front.append(ind)
    
    return pareto_front


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    individual_length = len(index_array)  # Length of the individual (example value)
    bounds = np.array([[-5, 5]] * individual_length)  # Example bounds for variables
    
    generations = 40
    population_size = 20
    pa = 0.25
    
    pareto_front = cuckoo_search_multiobjective(bounds, generations, population_size, pa)
    
    front = []
    for ind in pareto_front:
        print (evaluate(ind))
        front.append (evaluate(ind))

    ref_point = [30.0, 20.0, 1.0]
    hypervolume = calculate_hypervolume(front, ref_point)
    print("Hypervolume:", hypervolume)






















