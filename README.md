# Brain Tumor Detection using KNN, Naive Bayes, and SVM

## Project Overview

This project explores the detection of brain tumors through the analysis of Magnetic Resonance Imaging (MRI) images. It utilizes K-Nearest Neighbors (KNN), Naive Bayes (NB), and Support Vector Machine (SVM) algorithms. This README offers a concise comparison of these methods, emphasizing their effectiveness and suitability for medical image analysis tasks. For a more detailed project background and methodology, refer to the attached PDF document, 'Vadana_Ioan_Grigore_334AA_KNN&NB'.

# Source Code
knn_nb_stack.py: Implements a stacking ensemble model that combines KNN and NB, using Logistic Regression as the meta-learner. This approach aims to leverage the strengths of both classifiers to improve prediction accuracy.
svm_model.py: Contains the implementation of the SVM model, which is configured with a linear kernel to classify the processed MRI images.

# Training and Evaluation Process
# Both models were trained on a dataset of preprocessed and labeled MRI images. The following metrics were used to evaluate the models:

## Accuracy: Measures the overall correctness of the model.
## Precision: Indicates the proportion of positive identifications that were actually correct.
## Recall: Measures the proportion of actual positives that were correctly identified.
## F1 Score: Harmonic mean of precision and recall, providing a single metric to assess model performance.

## Usage

Place your dataset in two separate folders, one for tumor and another for no_tumor images.
Update the folder paths in the script:

tumor_images, tumor_labels = load_images_from_folder('path_to_tumor_images', 1)
no_tumor_images, no_tumor_labels = load_images_from_folder('path_to_no_tumor_images', 0)

Run the script to train and test the SVM model on your data.

# Results and Comparison

# KNN/NB Results
## Accuracy: Detailed accuracy figures are provided in the results section.
## Precision and Recall: Specific values for each class (tumor, no tumor) help understand the model's ability to distinguish and correctly classify images.
## F1 Score: Summarizes the balance between precision and recall.

# SVM Results
## Accuracy: Direct comparison with the KNN/NB model shows how SVM performs with a different algorithmic approach.
## Precision and Recall: Evaluated for the same classes, allowing for direct metric comparison.
## F1 Score: Provides insight into the SVM's overall efficiency in balancing recall and precision.

# Comparison
Performance: The SVM model typically offers strong performance on image classification tasks, especially with a well-tuned kernel. The stacking model (KNN/NB) attempts to improve prediction robustness by combining classifier predictions.
Complexity and Speed: SVMs can be computationally intensive, especially with large datasets and complex kernels, whereas KNN/NB, though simpler, might require substantial memory.
Suitability: Depending on the dataset characteristics (size, imbalance), one model may outperform the other. The detailed analysis explores these aspects, providing a guide on when to use each model effectively.
Visual Results: The provided graphs and performance charts (confusion matrices, ROC curves) visually underline the strengths and weaknesses of each approach, assisting in a clearer understanding of practical implications.

## Additional detailed comparisons and analysis can be found in the attached PDF document, 'Vadana_Ioan_Grigore_334AA_KNN&NB'. This document provides a comprehensive overview of the project findings and methodological approaches.

## Disclaimer
The performance of the model depends on the quality and size of the dataset. This script is intended for educational purposes and may require adjustments for production use.

# License

- This project is protected by copyright and is not available under any public license. All rights are reserved. No part of this project may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without prior written permission from the author.

- © 2024 Vadana Ioan-Grigore. All rights reserved.

# Contact
For support or to report issues, please email grigorevadana3@gmail.com
