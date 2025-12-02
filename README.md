# ORUGA: Optimizing Readability Using Genetic Algorithms

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.knosys.2023.111273-blue.svg)](https://doi.org/10.1016/j.knosys.2023.111273)
[![Journal](https://img.shields.io/badge/Journal-Knowledge--Based_Systems-orange.svg)](https://www.sciencedirect.com/journal/knowledge-based-systems)
[![PyGAD Version](https://img.shields.io/badge/PyGAD-2.1.0-red.svg)](https://pypi.org/project/pygad/2.1.0/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Official implementation of the paper published in *Knowledge-Based Systems* (2024).**

**ORUGA** (**O**ptimizing **R**eadability **U**sing **G**enetic **A**lgorithms) is an unsupervised framework designed to automatically enhance the readability of text. Unlike deep learning approaches that require massive training datasets, ORUGA uses evolutionary strategies (Genetic Algorithms) to minimize complexity metrics (like FKGL) while preserving semantic meaning.

![ORUGA Process Example](example.png)

## 📄 Citation
If you utilize this framework or code in your research, **please cite the following paper**:

```bibtex
@article{martinez2024oruga,
    author = {Jorge Martinez-Gil},
    title = {Optimizing readability using genetic algorithms},
    journal = {Knowledge-Based Systems},
    volume = {284},
    pages = {111273},
    year = {2024},
    issn = {0950-7051},
    doi = {10.1016/j.knosys.2023.111273}    
}
````

## 📚 Tutorials & Context

For a general audience overview of the concepts behind this framework, refer to this three-part series on Medium:

  * [Part 1: Introduction to Readability Optimization](https://medium.com/@jorgemarcc/readability-optimization-in-python-1-3-4491a5216cf0)
  * [Part 2: Implementation Details](https://medium.com/@jorgemarcc/readabilty-optimization-in-python-2-3-39a4bc4e98e)
  * [Part 3: Advanced Optimization Strategies](https://medium.com/@jorgemarcc/readability-optimization-in-python-3-3-7cbe204cafef)

## ⚙️ Installation

To reproduce the experiments, install the dependencies:

```bash
pip install -r requirements.txt
```

> [\!WARNING]
> **CRITICAL DEPENDENCY CONFLICT**
>
> 1.  **Package Name:** Ensure you use `pygad==2.1.0`. Newer versions may cause compatibility errors with the evolutionary logic.
> 2.  **Namespace Conflict:** There is a known namespace collision between `Readability` and `readability-lxml`.
>       * This project uses `py-readability-metrics`.
>       * **Do not** install `readability-lxml` in the same environment, or the imports will fail.

## 🧪 Experimental Reproduction

The repository allows you to reproduce the single-objective and multi-objective evolutionary experiments reported in the paper.

### 1\. Single-Objective Optimization

These scripts focus solely on minimizing the **FKGL (Flesch-Kincaid Grade Level)** score using different synonym replacement strategies.

| Script | Strategy | Description |
| :--- | :--- | :--- |
| `oruga_wordnet.py` | **WordNet** | Uses the NLTK WordNet lexical database for synonym retrieval. Fast and standard. |
| `oruga_word2vec.py` | **Word2Vec** | Uses vector embeddings to find synonyms. *Note: Slower execution due to vector operations.* |
| `oruga_webscraping.py` | **Web** | Scrapes external thesaurus sites. *Note: Please use responsibly to avoid rate limiting.* |

### 2\. Multi-Objective Optimization (NSGA-II & GDE3)

These scripts implement the advanced contributions of the paper, simultaneously minimizing **Readability Score (FKGL)** and **Text modification rate**, preventing the algorithm from changing too many words (Semantic Drift).

**Using NSGA-II (Non-dominated Sorting Genetic Algorithm II):**

```bash
# Basic Semantic Protection
python oruga2_nsga2.py

# Advanced Semantic Protection (using Word Mover's Distance - WMD)
python oruga2_nsga2_wmd.py
```

**Using GDE3 (Generalized Differential Evolution 3):**

```bash
# Basic Semantic Protection
python oruga2_gde3.py

# Advanced Semantic Protection (using Word Mover's Distance - WMD)
python oruga2_gde3_wmd.py
```

## 📊 Dataset

The repository includes `texts.txt`, which contains the benchmarking dataset used in the study:

  * **Content:** 10 text samples extracted from Wikipedia.
  * **Diversity:** Varies in length, topic, and initial complexity levels to test the robustness of the algorithm.

## 📄 License

This project is licensed under the **MIT License**.
