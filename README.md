# Outlier Detection in Categorical or Mixed Datasets

by Felippe P. Ferreira, and Robson L. F. Cordeiro

## Abstract

> ....

## Overview

This project provides a comprehensive survey and analysis of strategies for outlier detection in datasets that contain categorical or mixed (categorical and numerical) features. While most outlier detection algorithms are designed for numerical data, a significant portion of real-world datasets, such as 55% of those in the UCI repository, include categorical features. This research addresses the critical question: What is the most effective way to process categorical data to identify outliers?

The study evaluates three primary strategies through extensive experimentation on 47 diverse datasets using 14 different outlier detection algorithms:

1.  Directly applying algorithms designed for categorical data.
2.  Converting categorical features into numerical values before applying numerical algorithms.
3.  Removing categorical features entirely and using only numerical data.

Based on the findings, this work also introduces a predictive model that helps determine the most suitable algorithm type (categorical or numerical) for a new dataset.

## Directory Tree

A summary of the file structure can be found in the following directory tree.

```bash
CategoricalDatasets
├── files                  			\\ Main container for all project assets.
│   ├── code               			\\ Contains all source code and analysis scripts.
│   │	└── algorithms				\\ Source code of the algorithms used in the research.
│   │		├── java
│   │		└── python
│   │
│   │   └── based_experiments   		\\ Jupyter notebooks for specific analyses.
│   │       └── analyses					\\ Directory containing scripts for analyzing the results of outlier detection algorithm processing.
│   │       	├── BoxPlot.ipynb 			\\ Notebook for generating box plots of method performance.
│   │       	├── Compare_Results.py 		\\ Notebook for grouping the results of the algorithms on the selected datasets.
│   │       	├── Critical_Difference.py 	\\ Script for creating critical difference diagrams.
│   │       	├── PairPlot.ipynb 			\\ Notebook for generating pair plots for rank comparison.
│   │       	└── Ranking.ipynb 			\\ Notebook for calculating and saving algorithm rankings.
│   │       └── decision_tree 			\\ Notebook for calculating and saving algorithm rankings.
│   │           └── Decision_Tree.ipynb \\ Notebook for generating decision trees for numerical or categorical algorithm selection.
│   │       └── processing_detection	\\ Directory containing scripts for processing datasets using outlier detection algorithms.
│   │       	├── Executor.py 		\\ Scripts to coordinate the execution of algorithms
│   │       	├── Processing_number.py \\ Scripts for processing datasets with numerical algorithms
│   │       	└── Processing.py 		\\ Scripts for processing datasets with categorical algorithms

│   │   └── preprocessing    \\ Jupyter notebooks for prepare datasets before algorithms processing.
│   │
│   ├── database           			\\ Stores all datasets used in the experiments.
│   │   └── based_experiments      	\\ Contains pre-processed datasets.
│   │       ├── finance 			\\ Datasets from finance tasks.
│   │       ├── medicine     		\\ Datasets from medicine tasks.
│   │       ├── network_security    \\ Datasets from network_security tasks.
│   │       ├── not_grouped         \\ Other dataset source.
│   │       ├── sciency          	\\ Datasets from sciency tasks.
│   │       └── synthetic          	\\ Datasets from synthetic tasks.
│   │   └── prediction_model        \\ Datasets to decision tree train.
│   │
│   └── results            	\\ Stores all output files from the experiments.
│       └── decision_tree  	\\ Results from decision tree model.
│       └── experiments    	\\ Results corresponding to the notebooks in the code folder.
│           ├── algorithms 	\\ Results for each algorithm.
│           │   ├── AVF     \\ Results for the AVF algorithm.
│           │   ├── LOF     \\ Results for the LOF algorithm.
│           │   ├── KNN     \\ Results for the KNN algorithm.
│           │   └── ...     \\ Other algorithm result folders.
│           │
│           ├── conversion_methods \\ Boxplots from conversion methods.
│           │
│           └── plot       		\\ Generated plots and figures from the analyses.
│               ├── BOXPLOT  	\\ Output images for box plots.
│               ├── PAIRPLOT 	\\ Output images for pair plots.
│               └── ...       	\\ Other plot images.
│           │
│           └── tables       	\\ Generated tables from results of algorithms.
│
└── README.md               \\ Project overview, setup instructions, and documentation.
```

## Key Contributions

This research offers three main contributions to the field of outlier detection:

### 1. Comparison of Categorical Data Processing Strategies

We conducted a large-scale comparison to determine the best strategy for handling categorical features in outlier detection.

-   **Finding:** Algorithms designed for categorical data, particularly the **CBRW algorithm**, generally outperform numerical algorithms on mixed datasets.
-   **Exception:** Numerical algorithms like **iForest** and **KNN** can achieve superior results in specific contexts, such as on datasets with a low proportion of categorical features (e.g., up to 34%) or in domains like network security.
-   **Conclusion:** The characteristics of a dataset are a crucial factor in selecting the most effective outlier detection algorithm.

### 2. Predictive Model for Algorithm Selection

To guide researchers and practitioners, we developed a predictive model to recommend the best approach for new, unseen datasets.

-   **Model:** A **Decision Tree** was trained on the experimental results to classify whether a new dataset is better suited for a numerical or a categorical outlier detection algorithm.
-   **Performance:** The model was validated on a new set of datasets and achieved an **accuracy of 80%** in recommending the most appropriate algorithm type.

### 3. Evaluation of Categorical-to-Numerical Conversion Methods

Given the prevalence of numerical algorithms, we investigated the impact of different methods for converting categorical features into numerical values.

-   **Methods Tested:** The study compared four different conversion techniques alongside the alternative of simply removing the categorical features.
-   **Result:** The **Correspondence Analysis** method was found to yield the best results, positively influencing the effectiveness of numerical outlier detection algorithms.

## Execution Instructions

This repository contains the datasets used in the research. The initial versions of the datasets used are in the **files/datasets** directory, organized by context in which they were created. As an initial step to reproduce the experiments, it is necessary to run the preprocessing scripts from the **files/code/preprocessing** directory:

-   **Prepare_dataset.ipynb**
-   **Converter.ipynb**.

The first script applies data normalization, removes duplicate instances, and downsampling if the outlier instance rate exceeds **5%** of the data. Subsequently, the **Converter.ipynb** script creates versions of the datasets containing only numerical features, allowing numerical algorithms to utilize these datasets. After both scripts have run, the datasets can be applied to outlier detection algorithms.

Once you have the results from the outlier detection algorithms on the datasets in the **file/datasets** directory, which should be stored in the **files/results/experiments/algorithms** directory, separated according to the algorithm name.

For each .csv result file in the **files/results/experiments/algorithms** directory, there must be a structure with the following header:

```sh
dataset;parameter;algorithm;auc;r_precision;adj_r_precision;average_precision;adj_average_precision;max_f1;adj_max_f1
```

Each line in this file represents a selected configuration for the algorithm on a dataset. The metrics illustrate the average result of the algorithm on the dataset, and there are also adjusted versions of the metrics, considering the imbalance characteristics of the datasets. Examples of file patterns have been added to the AVF and DeepSVDD algorithm directories.

To grouping the results, the scripts in **files/code/experiments** should be executed in the following order:

-   **Compare_Results.ipynb**
-   **Ranking_Table.ipynb**
-   **Pair_Plot.ipynb**
-   **Box_Plot.ipynb**

The **Compare_Results.ipynb** file compiles the algorithm results and produces the CDD diagrams: general and clustered. From the combination of results, the **Ranking_Table.ipynb** file generates the ranking tables of the algorithms, presenting the average results of each one. The last two files, **Pair_Plot.ipynb** and **Box_Plot.ipynb**, produce the data representations as the file names suggest.

The plots in the **Pair_Plot.ipynb** file relate the results of the detection algorithms and the metrics used in this research (*AUC*, *P@n*, *Average Precision*, *Max-F1*). The boxplots illustrate the performance of the algorithms in relation to the strategies for converting categorical features into numerical features (*Correspondence Analysis*, *One-Hot Encoding*, *IDF* (Inverse Document Frequency), *Pivot-Based*, and removal of categorical features).

## Keywords

-   Categorical Dataset
-   Outlier Detection
-   Comparison of Approaches

## Authors

-   **Felippe Pires Ferreira**
    -   Institute of Mathematical and Computer Sciences, University of São Paulo (USP), São Carlos, SP, Brazil.
    -   *Email: [felippe_pires@usp.br](mailto:felippe_pires@usp.br)*
-   **Robson L. F. Cordeiro**
    -   Institute of Mathematical and Computer Sciences, University of São Paulo (USP), São Carlos, SP, Brazil.