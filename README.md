# Brain Tumor Detection using KNN and Naive Bayes

## Project Overview

This project addresses the detection of brain tumors by analyzing Magnetic Resonance Imaging (MRI) images using K-Nearest Neighbors (KNN) and Naive Bayes (NB) algorithms. Additionally, the outcomes of these methods are compared with results obtained using the Support Vector Machine (SVM) algorithm to evaluate the effectiveness of different approaches.

## Source Code

- `knn_nb_stack.py`: Implementation of the stacking algorithm combining KNN and NB with Logistic Regression as the meta-learner.
- `svm_model.py`: Source code for the SVM model implementation (provided separately).

## Training and Evaluation Process

The KNN/NB model was trained on a dataset consisting of preprocessed and labeled MRI images. The model's performance was evaluated based on accuracy, precision, recall, and F1 score. Similarly, the SVM model was evaluated using the same metrics to enable direct comparison.

## Results and Comparison

- **KNN/NB Results**: Details on the accuracy, precision, recall, and F1 score achieved by the KNN/NB model.
- **SVM Results**: Similar information about the performance of the SVM model.
- **Comparison**: An analysis of the performance of the two approaches, highlighting the advantages and limitations of each.

## Usage

To run the models and evaluate their performance:

```bash
python knn_nb_stack.py
python svm_model.py


This README provides an overview of the project, implementation details, as well as a structure for including results and comparisons. Please fill in the specific sections with information and results from your project.
