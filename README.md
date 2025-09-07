# Outlier Detection in Categorical or Mixed Datasets

## Overview

This project provides a comprehensive survey and analysis of strategies for outlier detection in datasets that contain categorical or mixed (categorical and numerical) features. While most outlier detection algorithms are designed for numerical data, a significant portion of real-world datasets, such as 55% of those in the UCI repository, include categorical features. This research addresses the critical question: What is the most effective way to process categorical data to identify outliers?

The study evaluates three primary strategies through extensive experimentation on 47 diverse datasets using 14 different outlier detection algorithms:
1.  Directly applying algorithms designed for categorical data.
2.  Converting categorical features into numerical values before applying numerical algorithms.
3.  Removing categorical features entirely and using only numerical data.

Based on the findings, this work also introduces a predictive model that helps determine the most suitable algorithm type (categorical or numerical) for a new dataset.

## Key Contributions

This research offers three main contributions to the field of outlier detection:

### 1. Comparison of Categorical Data Processing Strategies
We conducted a large-scale comparison to determine the best strategy for handling categorical features in outlier detection.

* **Finding:** Algorithms designed for categorical data, particularly the **CBRW algorithm**, generally outperform numerical algorithms on mixed datasets.
* **Exception:** Numerical algorithms like **iForest** and **KNN** can achieve superior results in specific contexts, such as on datasets with a low proportion of categorical features (e.g., up to 34%) or in domains like network security.
* **Conclusion:** The characteristics of a dataset are a crucial factor in selecting the most effective outlier detection algorithm.

### 2. Predictive Model for Algorithm Selection
To guide researchers and practitioners, we developed a predictive model to recommend the best approach for new, unseen datasets.

* **Model:** A **Decision Tree** was trained on the experimental results to classify whether a new dataset is better suited for a numerical or a categorical outlier detection algorithm.
* **Performance:** The model was validated on a new set of datasets and achieved an **accuracy of 80%** in recommending the most appropriate algorithm type.

### 3. Evaluation of Categorical-to-Numerical Conversion Methods
Given the prevalence of numerical algorithms, we investigated the impact of different methods for converting categorical features into numerical values.

* **Methods Tested:** The study compared four different conversion techniques alongside the alternative of simply removing the categorical features.
* **Result:** The **Correspondence Analysis** method was found to yield the best results, positively influencing the effectiveness of numerical outlier detection algorithms.

## Keywords
* Categorical Dataset
* Outlier Detection
* Comparison of Approaches

## Authors
* **Felippe Pires Ferreira**
    * Institute of Mathematical and Computer Sciences, University of São Paulo (USP), São Carlos, SP, Brazil.
    * *Email: felippe_pires@usp.br*
* **Robson L. F. Cordeiro**
    * School of Computer Science, Carnegie Mellon University (CMU), Pittsburgh, PA, USA.
    * *Email: robsonc@andrew.cmu.edu*
