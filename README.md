## Labs for CS5361 Machine Learning
This repository contains my submissions/labs for the Fall 2019 CS5361 Machine Learning Course. All resources and provided instructions are provided by the [instructor's site](http://www.cs.utep.edu/ofuentes/cs4361.html).

<details>
<summary>Lab 1: K-nearest Neighbors</summary>

The `lab1` directory contains the following files:
- `_instr.pdf` which contains the instructions for the assignment
- `knn.py` which is the provided script to modify for building a k-nearest-neighbors predictor model
- `mnist.py` which is the provided script to load the mnist dataset for training/testing
- `zeroR.py` which is the provided script containing a predictor model
- `report.py` which is my submitted report for this assignment
### To Run
1. Modify the `dir` variable in the `knn.py` script to direct the path to your downloaded datasets.
2. Run the `mnist.py` code first.
3. Run `knn.py`. See datasets for any additional files required to run the program.
### Datasets
The dataset used for this program is the MNIST dataset and the Solar particle dataset, as provided by the instructor on the course webpage.
### Results
Experimental results for this assignment can be found in [this Google sheets](http://bit.ly/19w_cs5361-results)* document, in the `lab1-knn` sheet.
<br>\*_This document may not be available after the course end date._ 
</details>

<details>
<summary>Lab 2: Decision Trees</summary>

The `lab2` directory contains the following files:
- `_instr.pdf` which contains the instructions for the assignment
- `magic04.txt` which is the provided dataset
- `decision_tree.py` which is the  provided script to modify for building a decision tree classification model
- `regression_tree.py` which is the provided script to modify for building a decision tree regression model
### To Run
1. Modify the `dir` variable in the `regression_tree.py` program to direct the path to your solar particle dataset.
2. Compile the `decision_tree.py` program or the `regression_tree.py` program, or both to your preference.
### Datasets
The dataset used for this program is provided by the instructor on the course webpage.
### Results
Experimental results for this assignment can be found in [this Google sheets](http://bit.ly/19w_cs5361-results)* document, in the `lab2-dectree` sheet.
<br>\*_This document may not be available after the course end date._
</details>

<details>
<summary>Lab 3: Decision and Regression Trees</summary>

The `lab3` directory contains the following files:
- `_instr.pdf` which contains the instructions for the assignment
- `decision_tree.py` which is the  provided script to modify for building a decision tree classification model
- `regression_tree.py` which is the provided script to modify for building a decision tree regression model
### To Run
1. Modify the `dir` variables in the `regression_tree.py` and `decision_tree.py` programs to direct the path to a dataset of your choice.
2. Compile the `decision_tree.py` program or the `regression_tree.py` program, or both to your preference.
### Datasets
The dataset used for this program is provided by the instructor on the course webpage.
### Results
Experimental results for this assignment can be found in [this Google sheets](http://bit.ly/19w_cs5361-results)* document, in the `lab2-dectree` sheet.
<br>\*_This document may not be available after the course end date._
</details>

<details>
<summary>Lab 4: The scikit library</summary>

The `lab4` directory contains the following files:
- `_instr.pdf` which contains the instructions for the assignment
- `__init__.py` which is the main script to compile the program
- `dataset.py` which contains the Dataset class that loads and stores the datasets for use in the program
- `dectree.py` which contains the classification and regressor predictor models for decision trees
- `forest.py` which contains the classification and regressor predictor models for forests
- `knn.py` which contains the classification and regressor predictor models for knn
- `logreg.py` which contains the classification and regressor predictor models for logistic regression
- `svm.py` which contains the classification and regressor predictor models for support vector machine <br>
For information about the other files in this directory, see the _Results_ section below.
### To Run
1. Modify the `dataset.py` script to access the dataset(s) of your choice
2. Compile the `__init__.py` program, or both to your preference.
### Datasets
The datasets used for this program are provided by the instructor on the course webpage.
### Results
Experimental results for this assignment can be found in the `res.txt` and `results.txt` files included in the `lab4` directory.
</details>

<details>
<summary>Lab 5: The keras library</summary>

The `lab5` directory contains the following files:
- `_instr.pdf` which contains the instructions for the assignment
- `__init__.py` which is the main script to compile the program
- `cnn.py` which contains the code to develop and test convolutional neural networks on two datasets: MNIST and CIFAR-10
- `dnn.py` which contains the code to develop and test fully connected dense neural networks on two datasets: solar particle and gamma ray
For information about the other files in this directory, see the _Results_ section below.
### To Run
1. Modify the `dataset.py` script to access the dataset(s) of your choice
2. Compile the `__init__.py` program, or both to your preference.
### Datasets
The datasets used for this program are either provided by the instructor on the course webpage (solar particle and gamma ray) or imported via the keras library (MNIST and CIFAR-10).
### Results
Experimental results for this assignment can be found in the `lab5\lab5.txt` file or, for specific runs, in the respective `lab5\results` directory containing `test##.txt` files.
</details>

<details>
<summary>Lab 6: Learning to Predict Sequences</summary>

The `lab6` directory contains the following files:
- `_instr.pdf` which contains the instructions for the assignment
- `__init__.py` which is the main script to compile the program
- `results-base.txt` which contains the accuracy results of the predictions by the baseline model
- `results-lstm.txt` which contains the accuracy results of the predictions by the LSTM model
- `results-conv.txt` which contains the accuracy results of the predictions by the convolutional model
For information about the other files in this directory, see the _Results_ section below.
### To Run
1. Modify the `dataset.py` script to access the dataset(s) of your choice.
2. Compile the `__init__.py` program.
### Datasets
The datasets used for this program are either provided by the instructor on the course webpage (solar dataset: `xrp.npy`).
### Results
Experimental results for this assignment can be found in the `npy` files. Accuracy results are presented in the `results-XXXX.txt` files.
</details>

## Final Project for CS5361 Machine Learning
<br>
<details>
    <summary>Predicting Personality through Text</summary>
   
The `project` directory contains the following files:
- `baseline.py` which contains the code to build and train the [baseline model](http://web.stanford.edu/class/archive/cs/cs221/cs221.1192/2018/restricted/posters/dkedia/poster.pdf)
- `main.py`  which contains the code to build and train the Naive-Bayes classifiers
- `doc2vec.py` which contains the code to build a Doc2Vec embeddings for the dataset
- `dataset.py` which contains the code to read and load the [dataset](https://www.kaggle.com/datasnaek/mbti-type) 
### To Run
1. For the first run (in both `baseline.py` and `main.py`), modify the constructor to _Dataset_ to contain `first_time=False`. This will load the data in your system (make sure _d_ in `dataset.py` reflects your system configuration) and create _npy_ files for NumPy to use in future runs.
2. In the second run (in both `baseline.py` and `main.py`), remove the `first_time=False` modification we used in (1). Then, change the main method to call `second_run(...)`.
3. In any future runs (in both `baseline.py` and `main.py`), comment out the line of code in (2), and run as normally.
</details>
