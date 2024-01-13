import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

# Helper Functions
def load_and_preprocess_image(image_path, size=(64, 64)):
    image = imread(image_path)
    image = rgb2gray(image)
    image = resize(image, size)
    return image / 255.0

def load_images_from_folder(folder, label, size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img = load_and_preprocess_image(os.path.join(folder, filename), size)
            images.append(img.flatten())
            labels.append(label)
    return images, labels

# Data Loading
no_tumor_dir = r'C:\Users\Grig\PycharmProjects\ML\Data\no_tumor'  # Replace with your path
tumor_dir = r'C:\Users\Grig\PycharmProjects\ML\Data\tumor'        # Replace with your path

no_tumor_images, no_tumor_labels = load_images_from_folder(no_tumor_dir, label=0)
tumor_images, tumor_labels = load_images_from_folder(tumor_dir, label=1)

images = np.array(no_tumor_images + tumor_images)
labels = np.array(no_tumor_labels + tumor_labels)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Base Learners
base_learners = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('nb', GaussianNB())
]

# Meta-Learner
meta_learner = LogisticRegression()

# Stacking Ensemble
stacked_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)

# Training the Stacked Model
stacked_model.fit(X_train, y_train)

# Training Evaluation
train_predictions = stacked_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)
print("Training Classification Report:")
print(classification_report(y_train, train_predictions))

# Making Predictions with the Stacked Model
stacked_predictions = stacked_model.predict(X_test)
stacked_proba = stacked_model.predict_proba(X_test)[:, 1]

# Evaluating the Stacked Model
print("\nStacked Model Test Classification Report:")
print(classification_report(y_test, stacked_predictions))

stacked_roc_auc = auc(*roc_curve(y_test, stacked_proba)[:2])
print("Stacked Model Test ROC AUC Score:", stacked_roc_auc)

# Plotting ROC Curve for the Stacked Model
plt.figure()
fpr, tpr, _ = roc_curve(y_test, stacked_proba)
plt.plot(fpr, tpr, label=f'Stacked Test (AUC = {stacked_roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Stacked Model')
plt.legend(loc="lower right")
plt.show()
