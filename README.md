# ORUGA: Optimizing Readability Using Genetic Algorithms

This repository contains code for reproducing the experiments reported in the paper
- Jorge Martinez-Gil: Optimizing readability using genetic algorithms. Knowl. Based Syst. 284: 111273 (2024)(https://doi.org/10.1016/j.knosys.2023.111273)

Also available is a collection of three articles on Medium that are written for a general audience.
- [[Readability Optimization in Python (1/3)]](https://medium.com/@jorgemarcc/readability-optimization-in-python-1-3-4491a5216cf0)
- [[Readability Optimization in Python (2/3)]](https://medium.com/@jorgemarcc/readabilty-optimization-in-python-2-3-39a4bc4e98e)
- [[Readability Optimization in Python (3/3)]](https://medium.com/@jorgemarcc/readability-optimization-in-python-3-3-7cbe204cafef)

# 🌍 Overview 

ORUGA is a method that seeks to optimize the readability of any text automatically. It is based on genetic algorithms, so it is unsupervised and requires no training.

![Example](example.png)

# 🛠️ Install
``` pip install -r requirements.txt```

(Important!! Please be very careful with the incompabilities of packages **Readability** and **readability-lxml**. The import Readability belongs to **py-readability-metrics**. Also, be sure to use the package pygad==2.1.0, otherwise, you will find errors.) 

# 📊 Dataset
```texts.txt```
Ten texts extracted from Wikipedia with different lengths, topics and readability levels.

# 🚀 Use
``` python oruga_wordnet.py```
Run the basic ORUGA program looking to minimize the FKGL score using the WordNet synonym library.

``` python oruga_word2vec.py```
Run the basic ORUGA program looking to minimize the FKGL score using the word2vec synonym method. (Slow)

``` python oruga_webscraping.py```
Run the basic ORUGA program looking to minimize the FKGL score using webscraping. (Please be responsible)

``` python oruga2_nsga2.py```
Run the second version of ORUGA using NSGA-II to minimize FKGL score and number of words to be replaced simultaneously.

``` python oruga2_gde3.py```
Run the second version of ORUGA using GDE3 to minimize FKGL score and number of words to be replaced simultaneously.

``` python oruga2_nsga2_wmd.py```
Run the third version of ORUGA using NSGA-II to minimize FKGL score, the number of words to be replaced simultaneously, and WMD for semantic distance.

``` python oruga2_gde3_wmd.py```
Run the third version of ORUGA using GDE3 to minimize FKGL score and number of words to be replaced simultaneously, and WMD for semantic distance.
 
## 📚 Citation
If you use ORUGA, please cite:

```
@article{martinez2024oruga,
	author = {Jorge Martinez-Gil},
	title = {Optimizing readability using genetic algorithms},
	journal = {Knowledge-Based Systems},
	volume = {284},
	pages = {111273},
	year = {2024},
	issn = {0950-7051},
	doi = {https://doi.org/10.1016/j.knosys.2023.111273}	
}

```
  
## 📄 License
MIT