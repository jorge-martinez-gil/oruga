# ORUGA
Optimizing Readability Using Genetic Algorithms
 
ORUGA is a method that seeks to optimize the readability of any text automatically. It is based on genetic algorithms, so it is unsupervised and requires no training.

![Example](example.png)

# Install
``` pip install -r requirements.txt```

# Dataset
```texts.txt```
Ten texts extracted from Wikipedia with different lengths, topics and readability levels.

# Use
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
 
## Citation
If you use ORUGA, please cite:

```
@inproceedings{martinez2023,

}

```
  
## License
MIT